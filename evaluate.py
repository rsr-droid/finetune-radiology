import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from pycocoevalcap.cider.cider import Cider
from rouge import Rouge
from chexbert import CheXbertMetrics
from cxr_bert import CXRBERT
import argparse

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model_and_processor(finetuned_model, original_model_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    
    processor = AutoProcessor.from_pretrained(original_model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        finetuned_model,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )
    
    return model, processor

def generate_description(model, processor, image_paths, prompt): # may need to update this for other conversation templates for different models (only set up for llava models atm)
    images = [Image.open(img_path).convert("RGB") for img_path in image_paths]

    # if path is a string, put it in a list
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    # chat template in interleaved format work same as in sampling videos. Just pass in as many images you want for a prompt
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": "Provide a description of the findings and impressions in the radiology images given the following images of the study."},
            ]+ [{"type": "image"} for _ in image_paths]
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, padding=True, return_tensors="pt").to(model.device)
    
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if key in ['input_ids', 'attention_mask']:
                inputs[key] = value.long()
            elif key == 'pixel_values':
                inputs[key] = value.half()
    
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_k=1,
        top_p=0.9,
    )
    return processor.decode(output[0], skip_special_tokens=True)

def extract_assistant_response(generated_output):
    parts = generated_output.split('assistant')
    if len(parts) > 1:
        return parts[-1].strip()
    else:
        return generated_output.strip()

def extract_study_id(image_path):
    # Split the path and find the component starting with 's'
    path_components = image_path.split('/')
    for component in path_components:
        if component.startswith('s') and component[1:].isdigit():
            return component
    return None  # Return None if no valid study_id is found

def generate_outputs(model, processor, test_set, num_samples=None):
    """Generate both descriptions and labels for the test set"""
    if num_samples is not None:
        test_set = test_set[:num_samples]
        print(f"Evaluating on {num_samples} samples")
    else:
        print(f"Evaluating on all {len(test_set)} samples")
    
    results = []
    print("Generating descriptions and labels...")
    for item in tqdm(test_set):
        study_id = extract_study_id(item['image'][0])
        if study_id is None:
            print(f"Warning: Could not extract study_id from {item['image'][0]}")
            continue
        
        gt_report = item['conversations'][1]['value']
        
        # Generate description
        generated_description = generate_description(
            model, processor, item['image'], 
            "Provide a description of the findings and impressions in the radiology images given the following images of the study."
        )
        generated_description = extract_assistant_response(generated_description)
        
        # Generate labels
        label_output = generate_description(
            model, processor, item['image'], 
            "What are the key radiological findings in this chest X-ray study?"
        )
        label_output = extract_assistant_response(label_output)
        
        results.append({
            'study_id': study_id,
            'ground_truth': gt_report,
            'generated_description': generated_description,
            'generated_labels': label_output
        })
    
    return results

def evaluate_lexical_metrics(results):
    """Calculate lexical metrics (BLEU, CIDEr, ROUGE)"""
    references = []
    hypotheses = []
    cider_gts = {}
    cider_res = {}
    
    print("Calculating lexical metrics...")
    print("Preprocessing data...")
    for i, result in enumerate(tqdm(results)):
        reference = nltk.word_tokenize(result['ground_truth'])
        hypothesis = nltk.word_tokenize(result['generated_description'])
        references.append([reference])
        hypotheses.append(hypothesis)
        
        cider_gts[i] = [result['ground_truth']]
        cider_res[i] = [result['generated_description']]
    
    print("Calculating BLEU scores...")
    smoothie = SmoothingFunction().method1
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    bleu_4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    
    print("Calculating CIDEr score...")
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(cider_gts, cider_res)
    
    print("Calculating ROUGE scores...")
    rouge_scorer = Rouge()
    rouge_scores = rouge_scorer.get_scores(
        [r['generated_description'] for r in results], 
        [r['ground_truth'] for r in results], 
        avg=True
    )
    
    return {
        'BLEU-1': bleu_1,
        'BLEU-2': bleu_2,
        'BLEU-3': bleu_3,
        'BLEU-4': bleu_4,
        'CIDEr': cider_score,
        'ROUGE-1': rouge_scores['rouge-1']['f'],
        'ROUGE-2': rouge_scores['rouge-2']['f'],
        'ROUGE-L': rouge_scores['rouge-l']['f']
    }

def evaluate_chexbert_metrics(results):
    """Calculate CheXbert metrics for label classification"""
    print("Calculating CheXbert metrics...")
    
    # Initialize CheXbert
    chexbert = CheXbertMetrics(
        bert_path='bert-base-uncased',
        checkpoint_path='/working/rajan/multiview-llm/Models/Single/cvt2distilgpt2/checkpoints/stanford/chexbert/chexbert.pth',
        ckpt_dir='ckpt',
        mbatch_size=1,
        exp_dir='metrics',
    )
    
    generated_labels = [r['generated_labels'] for r in results]
    ground_truth = [r['ground_truth'] for r in results]
    
    chexbert.update(generated_labels, ground_truth, list(range(len(generated_labels))))
    metrics = chexbert.compute()
    
    # Convert tensor values to float
    return {k: v.item() if isinstance(v, torch.Tensor) else v 
            for k, v in metrics.items()}

def evaluate_cxr_bert_metrics(results):
    """Calculate CXR-BERT similarity metrics"""
    print("Calculating CXR-BERT metrics...")
    cxr_bert_metric = CXRBERT(
        ckpt_dir='ckpt', 
        mbatch_size=10, 
        exp_dir='metrics',
        split='test',
        accumulate_over_dicoms=False
    )

    # Prepare inputs for CXR-BERT
    generated_descriptions = [r['generated_description'] for r in results]
    ground_truth = [r['ground_truth'] for r in results]
    study_ids = [r['study_id'] for r in results]

    # Update and compute CXR-BERT metrics
    cxr_bert_metric.update(generated_descriptions, [[gt] for gt in ground_truth], study_ids)
    metrics = cxr_bert_metric.compute(epoch=1)

    # Convert tensor values to float
    return {k: v.item() if isinstance(v, torch.Tensor) else v 
            for k, v in metrics.items()}


def save_results_and_metrics(results, metrics, model_path, output_dir):
    """Save results and metrics to files"""
    model_name = os.path.basename(model_path)
    test_output_dir = os.path.join(output_dir, 'test_outputs')
    metrics_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(test_output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save results
    results_path = os.path.join(test_output_dir, f'{model_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save metrics
    metrics_path = os.path.join(metrics_dir, f'{model_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return results_path, metrics_path

def evaluate_test_set(test_set_path, finetuned_model, original_model_id, output_dir, num_samples=None):
    """Main evaluation function"""
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load model and processor
    model, processor = load_model_and_processor(finetuned_model, original_model_id)
    
    # Load test set
    with open(test_set_path, 'r') as f:
        test_set = json.load(f)
    
    # Generate outputs
    results = generate_outputs(model, processor, test_set, num_samples)
    
    # Calculate metrics
    lexical_metrics = evaluate_lexical_metrics(results)
    chexbert_metrics = evaluate_chexbert_metrics(results)
    cxr_bert_metrics = evaluate_cxr_bert_metrics(results)
    
    # Combine metrics
    all_metrics = {
        'lexical_metrics': lexical_metrics,
        'classification_metrics': chexbert_metrics,
        'semantic_metrics': cxr_bert_metrics
    }
    
    # Save results and metrics
    results_path, metrics_path = save_results_and_metrics(results, all_metrics, finetuned_model, output_dir)
    
    print(f"Evaluation complete. Results saved to {results_path}")
    print(f"Metrics saved to {metrics_path}")
    
    # Print summary of metrics
    print("\n===== Metrics Summary =====")
    print("\nLexical Metrics:")
    for key, value in all_metrics['lexical_metrics'].items():
        print(f"  {key}: {value:.3f}")
    
    print("\nClassification Metrics:")
    for key, value in all_metrics['classification_metrics'].items():
        print(f"  {key}: {value:.3f}")
    
    print("\nSemantic Metrics (CXR-BERT):")
    for key, value in all_metrics['semantic_metrics'].items():
        print(f"  {key}: {value:.3f}")
    print("==================================\n")

if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model on the test set.")
    parser.add_argument("--test_set_path", type=str, required=True, help="Path to the test set JSON file")
    parser.add_argument("--finetuned_model", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--original_model_id", type=str, default="llava-hf/llava-interleave-qwen-7b-hf", 
                       help="ID of the original model")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (None for all)")

    args = parser.parse_args()
    evaluate_test_set(args.test_set_path, args.finetuned_model, args.original_model_id, args.output_dir, args.num_samples)


# export CUDA_VISIBLE_DEVICES=2 
# /home/rajan/.conda/envs/lmms-finetune/bin/python evaluate.py --test_set_path /working/rajan/multiview-llm/Models/Finetune/dataset/data/json/multi/no_labels/mimic_cxr_multi_test_findings.json \
#                    --finetuned_model /working/rajan/multiview-llm/Models/Finetune/lmms-finetune/checkpoints/llava-interleave-qwen-7b_lora-True_qlora-True \
#                    --original_model_id llava-hf/llava-interleave-qwen-7b-hf \
#                    --output_dir outputs \
#                    --num_samples 2
    

