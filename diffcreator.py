import torch
import os
from tqdm import tqdm
import logging
from safetensors import safe_open
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define model paths
hermes_model_path = "C:\\Users\\artno\\.cache\\huggingface\\hub\\models--NousResearch--Hermes-3-Llama-3.1-8B\\snapshots\\aabb745a717e133b74dcae23195d2635cf5f38cc"
llama_model_path = "c:\\Users\\artno\\.cache\\huggingface\\hub\\models--meta-llama--Meta-Llama-3.1-8B-Instruct\\snapshots\\5206a32e0bd3067aef1ce90f5528ade7d866253f"

# Create the model_diff directory
os.makedirs("model_diff", exist_ok=True)

def get_model_files(model_path):
    logging.info(f"Checking model files in: {model_path}")
    safetensor_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
    if not safetensor_files:
        logging.warning(f"No .safetensors files found in {model_path}. Contents: {os.listdir(model_path)}")
        raise ValueError(f"No .safetensors files found for {model_path}")
    logging.info(f"Found {len(safetensor_files)} .safetensors files")
    return safetensor_files

def create_unique_filename(base_name):
    counter = 0
    while True:
        filename = f"{base_name}_{counter}.pt" if counter > 0 else f"{base_name}.pt"
        if not os.path.exists(os.path.join("model_diff", filename)):
            return filename
        counter += 1

def compare_layers(llama_path, hermes_path, llama_file, hermes_file):
    logging.info(f"Comparing {llama_file} and {hermes_file}")
    with safe_open(os.path.join(llama_path, llama_file), framework="pt", device="cpu") as llama_state:
        with safe_open(os.path.join(hermes_path, hermes_file), framework="pt", device="cpu") as hermes_state:
            for name in tqdm(llama_state.keys(), desc=f"Comparing {llama_file}"):
                if name in hermes_state.keys():
                    llama_tensor = llama_state.get_tensor(name).to(torch.float32)
                    hermes_tensor = hermes_state.get_tensor(name).to(torch.float32)
                    diff = hermes_tensor - llama_tensor
                    safe_name = name.replace('.', '_').replace(':', '_')
                    filename = create_unique_filename(safe_name)
                    torch.save(diff, os.path.join("model_diff", filename))
                    logging.info(f"Saved difference for {name} as {filename}")
                else:
                    logging.warning(f"Tensor {name} not found in Hermes model")

def process_model_shards():
    llama_files = get_model_files(llama_model_path)
    hermes_files = get_model_files(hermes_model_path)

    if len(llama_files) != len(hermes_files):
        logging.warning(f"Models have different numbers of parameter files: Llama ({len(llama_files)}), Hermes ({len(hermes_files)})")

    # Sort files to ensure correct pairing
    llama_files.sort()
    hermes_files.sort()

    for llama_file, hermes_file in zip(llama_files, hermes_files):
        compare_layers(llama_model_path, hermes_model_path, llama_file, hermes_file)

def main():
    try:
        process_model_shards()
        logging.info("Model difference analysis complete. Differences saved in the 'model_diff' directory.")
        
        # Create a mapping file
        mapping = {}
        for filename in os.listdir("model_diff"):
            if filename.endswith('.pt'):
                original_name = filename.replace('_', '.').replace('.pt', '')
                mapping[original_name] = filename
        
        with open(os.path.join("model_diff", "diff_mapping.json"), 'w') as f:
            json.dump(mapping, f, indent=2)
        logging.info("Created diff_mapping.json in the 'model_diff' directory.")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()