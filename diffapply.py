import torch
from safetensors.torch import load_file, save_file
import os
import json
import logging
from tqdm import tqdm
import shutil
import argparse
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
llama_model_path = r"c:\Users\artno\.cache\huggingface\hub\models--meta-llama--Meta-Llama-3.1-8B-Instruct\snapshots\5206a32e0bd3067aef1ce90f5528ade7d866253f"
hermes_model_path = r"C:\Users\artno\.cache\huggingface\hub\models--NousResearch--Hermes-3-Llama-3.1-8B\snapshots\aabb745a717e133b74dcae23195d2635cf5f38cc"
diff_dir = "model_diff"
optimized_diff_dir = "optimized_diff"
new_model_path = "New_model"

def apply_update(param, update):
    if isinstance(update, tuple):  # Low-rank update
        U, V = update
        if param.dim() == 1:
            return param + U
        else:
            return param + torch.mm(U, V)
    else:  # Full update
        return param + update

def calculate_mse(tensor1, tensor2):
    return torch.nn.functional.mse_loss(tensor1, tensor2).item()

def process_shard(shard_file, weight_map, diff_dir, optimized_diff_dir, hermes_shard_path, diff_type_to_save):
    logging.info(f"Processing shard: {shard_file}")
    llama_shard_path = os.path.join(llama_model_path, shard_file)
    llama_tensors = load_file(llama_shard_path)
    hermes_tensors = load_file(hermes_shard_path)
    
    modified_tensors = {}
    
    for tensor_name, tensor in tqdm(llama_tensors.items(), desc="Processing tensors"):
        tensor = tensor.to(torch.float32)
        
        # Process original diff
        original_diff_file = os.path.join(diff_dir, f"{tensor_name.replace('.', '_')}.pt")
        if os.path.exists(original_diff_file):
            original_update = torch.load(original_diff_file, map_location="cpu").to(torch.float32)
            original_updated_tensor = apply_update(tensor, original_update)
            original_mse = calculate_mse(original_updated_tensor, hermes_tensors[tensor_name])
        else:
            original_mse = None
        
        # Process optimized diff
        optimized_diff_file = os.path.join(optimized_diff_dir, f"{tensor_name.replace('.', '_')}_lowrank.pt")
        if os.path.exists(optimized_diff_file):
            U, V = torch.load(optimized_diff_file, map_location="cpu")
            optimized_update = (U.to(torch.float32), V.to(torch.float32))
            optimized_updated_tensor = apply_update(tensor, optimized_update)
            optimized_mse = calculate_mse(optimized_updated_tensor, hermes_tensors[tensor_name])
        else:
            optimized_mse = None
        
        # Log the MSE for this tensor
        logging.info(f"Tensor: {tensor_name}")
        logging.info(f"  Original diff MSE: {original_mse}")
        logging.info(f"  Optimized diff MSE: {optimized_mse}")
        
        # Save the appropriate tensor based on diff_type_to_save
        if diff_type_to_save == 'original' and original_mse is not None:
            modified_tensors[tensor_name] = original_updated_tensor
        elif diff_type_to_save == 'optimized' and optimized_mse is not None:
            modified_tensors[tensor_name] = optimized_updated_tensor
        else:
            modified_tensors[tensor_name] = tensor
    
    # Save the modified shard if required
    if diff_type_to_save:
        output_path = os.path.join(new_model_path, shard_file)
        save_file(modified_tensors, output_path, metadata={"format": "pt"})
    
    # Clear memory
    del llama_tensors, hermes_tensors, modified_tensors
    gc.collect()
    torch.cuda.empty_cache()

def main(diff_type_to_save):
    os.makedirs(new_model_path, exist_ok=True)
    
    # Load the index file for Llama
    with open(os.path.join(llama_model_path, "model.safetensors.index.json"), 'r') as f:
        llama_index = json.load(f)
    
    # Load the index file for Hermes
    with open(os.path.join(hermes_model_path, "model.safetensors.index.json"), 'r') as f:
        hermes_index = json.load(f)
    
    # Process each shard
    for shard_file in set(llama_index['weight_map'].values()):
        hermes_shard_file = list(set(hermes_index['weight_map'].values()))[list(set(llama_index['weight_map'].values())).index(shard_file)]
        hermes_shard_path = os.path.join(hermes_model_path, hermes_shard_file)
        
        process_shard(shard_file, llama_index['weight_map'], diff_dir, optimized_diff_dir, hermes_shard_path, diff_type_to_save)
    
    # Copy necessary files if we're saving a new model
    if diff_type_to_save:
        shutil.copy2(
            os.path.join(llama_model_path, "model.safetensors.index.json"),
            os.path.join(new_model_path, "model.safetensors.index.json")
        )
        shutil.copy2(
            os.path.join(llama_model_path, "config.json"),
            os.path.join(new_model_path, "config.json")
        )
        for file in os.listdir(llama_model_path):
            if file.startswith("tokenizer") or file.endswith("tokenizer.json"):
                shutil.copy2(
                    os.path.join(llama_model_path, file),
                    os.path.join(new_model_path, file)
                )
        
        logging.info(f"Model processing complete. New model saved with applied {diff_type_to_save} updates.")
    else:
        logging.info("Diff comparison complete. No new model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply and compare diffs to Llama model")
    parser.add_argument("--save", choices=['original', 'optimized', 'none'], default='none',
                        help="Which diff to apply when saving the new model (default: none)")
    args = parser.parse_args()
    
    main(args.save if args.save != 'none' else None)