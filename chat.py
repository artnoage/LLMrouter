import torch
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
import os
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model paths
new_model_path = "New_model"  # Path to the new 32-bit model we created
llama_model_path = "c:\\Users\\artno\\.cache\\huggingface\\hub\\models--meta-llama--Meta-Llama-3.1-8B-Instruct\\snapshots\\5206a32e0bd3067aef1ce90f5528ade7d866253f"


# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(llama_model_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

def check_model_files():
    required_files = ['config.json', 'model.safetensors.index.json']
    for file in required_files:
        file_path = os.path.join(new_model_path, file)
        if not os.path.exists(file_path):
            logging.error(f"Required file {file} not found in {new_model_path}")
            return False
    
    # Check for at least one .safetensors file
    safetensor_files = [f for f in os.listdir(new_model_path) if f.endswith('.safetensors')]
    if not safetensor_files:
        logging.error(f"No .safetensors files found in {new_model_path}")
        return False
    
    logging.info(f"Found {len(safetensor_files)} .safetensors files")
    return True

def load_model():
    logging.info("Loading the model. This may take a few moments...")
    
    if not check_model_files():
        return None
    
    try:
        # Load and log config
        with open(os.path.join(new_model_path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Set up 8-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )

        # Load the model with 8-bit quantization
        model = LlamaForCausalLM.from_pretrained(
            new_model_path,
            quantization_config=quantization_config,
            device_map='auto',
            low_cpu_mem_usage=True,
        )

        model.eval()
        logging.info("Model loaded successfully with 8-bit quantization!")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}", exc_info=True)
        return None

def generate_response(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(generated_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}", exc_info=True)
        return ""

def chat():
    model = load_model()
    if model is None:
        logging.error("Failed to load the model. Exiting.")
        return

    print("Welcome to the chat! Type 'quit' to exit.")
    
    conversation_history = "A chat between a human and an AI assistant.\n\n"
    while True:
        user_input = input("Human: ")
        if user_input.lower() == 'quit':
            break
        
        conversation_history += f"Human: {user_input}\nAssistant: "
        
        response = generate_response(model, conversation_history)
        print("Assistant:", response)
        
        conversation_history += f"{response}\n"

if __name__ == "__main__":
    logging.info(f"Running chat on device: {device}")
    chat()