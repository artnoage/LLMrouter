import torch
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets
import gc
import time
import psutil
import GPUtil
from tqdm import tqdm
import os
import argparse
import numpy as np
import flash_attn

# Set the device
device = torch.device("cuda")

# Define model paths
llama_model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
hermes_model_path = "NousResearch/Hermes-3-Llama-3.1-8B"
new_model_path = "New_model"

# Define the quantization config
quantization_config = BitsAndBytesConfig(load_in_8bit=True) if device.type == "cuda" else None

def print_gpu_utilization():
    gpus = GPUtil.getGPUs()
    for i, gpu in enumerate(gpus):
        print(f'GPU {i}: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB')
    return gpus[0].memoryUsed if gpus else 0

def print_summary(stage):
    print(f"\n=== {stage} ===")
    gpu_mem = print_gpu_utilization()
    print(f"CPU RAM Free: {psutil.virtual_memory().available / 1024 ** 3:.2f} GB")
    return gpu_mem

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return print_summary("After memory clearing")

def load_model_and_tokenizer(model_path):
    print_summary("Before model loading")
    try:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Set the pad token to the eos token
        tokenizer.pad_token = tokenizer.eos_token
        
        print_summary("After model loading")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        print_summary("After failed model loading")
        return None, None

def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1]  # Get logits of the last token
        
        # Calculate probabilities for all tokens
        probs = torch.nn.functional.softmax(logits, dim=0)
        
        # Get probabilities and ranks for A, B, C, D
        answer_probs = []
        answer_ranks = []
        for ans in ["A", "B", "C", "D"]:
            ans_token_id = tokenizer.encode(ans, add_special_tokens=False)[0]
            ans_prob = probs[ans_token_id].item()
            ans_rank = (probs > probs[ans_token_id]).sum().item() + 1
            answer_probs.append(ans_prob)
            answer_ranks.append(ans_rank)
        
        # Get the most probable answer among A, B, C, D
        pred = ["A", "B", "C", "D"][np.argmax(answer_probs)]
        
        # Get the top 10 most likely tokens
        top_probs, top_indices = torch.topk(probs, 10)
        top_tokens = [tokenizer.decode([idx]) for idx in top_indices]
        
    return pred, answer_probs, answer_ranks, top_tokens, top_probs.tolist()

def run_benchmark(model, tokenizer, dataset, dataset_name, model_name, percentage):
    num_samples = max(1, int(len(dataset) * (percentage / 100.0)))
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(num_samples))
    
    results = []
    all_probs = []
    
    for i, item in enumerate(tqdm(dataset, desc=f"Processing {dataset_name} samples", unit="sample")):
        if dataset_name == "MMLU":
            choices_text = ", ".join(item['choices'])
            prompt = f"Question: {item['question']}\nChoices: {choices_text}\nAnswer:"
            correct_letter = ["A", "B", "C", "D"][int(item['answer'])]
        else:  # ARC-E
            choices_text = "\n".join([f"{label}. {text}" for label, text in zip(item['choices']['label'], item['choices']['text'])])
            prompt = f"Question: {item['question']}\nChoices:\n{choices_text}\nAnswer:"
            correct_letter = item['answerKey']
        
        pred, probs, ranks, top_tokens, top_probs = generate_response(model, tokenizer, prompt)
        
        is_correct = int(pred == correct_letter)
        results.append(is_correct)
        all_probs.append(probs)
        
        print(f"\nSample {i+1}:")
        print(f"Correct Answer: {correct_letter}")
        print(f"Predicted Answer: {pred}")
        print(f"Probabilities: A: {probs[0]:.6f}, B: {probs[1]:.6f}, C: {probs[2]:.6f}, D: {probs[3]:.6f}")
        print(f"Ranks: A: {ranks[0]}, B: {ranks[1]}, C: {ranks[2]}, D: {ranks[3]}")
        print(f"Top 10 tokens and their probabilities:")
        for token, prob in zip(top_tokens, top_probs):
            print(f"  {token}: {prob:.6f}")
        print(f"Is Correct: {is_correct}")
    
    accuracy = sum(results) / len(results)
    print(f"\n{dataset_name} - {model_name} - Total correct: {sum(results)}/{len(results)}")
    print(f"{dataset_name} - {model_name} - Accuracy: {accuracy:.2%}")
    return accuracy, np.array(all_probs)

def main(args):
    print(f"Running inference on device: {device}")
    print(f"Using {args.percentage}% of the dataset")
    
    # Load datasets
    print("Loading datasets...")
    mmlu_dataset = load_dataset("cais/mmlu", "all")
    mmlu_dataset = concatenate_datasets([mmlu_dataset[split] for split in mmlu_dataset.keys()])
    
    arc_e_dataset = load_dataset("ai2_arc", "ARC-Easy")
    arc_e_dataset = concatenate_datasets([arc_e_dataset[split] for split in arc_e_dataset.keys()])
    
    print(f"MMLU dataset loaded. Total datapoints: {len(mmlu_dataset)}")
    print(f"ARC-E dataset loaded. Total datapoints: {len(arc_e_dataset)}")

    models = [
        ("Llama", llama_model_path),
        ("Hermes", hermes_model_path),
        ("New_model", new_model_path)
    ]

    datasets = [
        ("MMLU", mmlu_dataset),
        ("ARC-E", arc_e_dataset)
    ]

    for model_name, model_path in models:
        print(f"\nBenchmarking {model_name} model:")
        model, tokenizer = load_model_and_tokenizer(model_path)
        if model is not None and tokenizer is not None:
            print(f"{model_name} model loaded.")
            for dataset_name, dataset in datasets:
                run_benchmark(model, tokenizer, dataset, dataset_name, model_name, args.percentage)
            del model
            clear_memory()
        else:
            print(f"Failed to load {model_name} model. Skipping benchmarks.")

    print("\nBenchmarking complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on models")
    parser.add_argument("--percentage", type=float, default=10.0, help="Percentage of the dataset to use (default: 10.0)")
    args = parser.parse_args()
    
    main(args)