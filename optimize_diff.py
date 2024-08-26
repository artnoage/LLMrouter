import torch
import os
import json
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_low_rank_tensor(original_tensor, rank_percentage=0.1, device='cuda'):
    logging.info(f"Original tensor shape: {original_tensor.shape}")
    
    if original_tensor.dim() == 1:
        # For 1D tensors, we'll just return the original tensor as U and a tensor of ones as V
        U = original_tensor.clone()
        V = torch.ones(1, 1, device=device)
        logging.info(f"1D tensor: U shape: {U.shape}, V shape: {V.shape}")
        return U, V
    
    min_dim = min(original_tensor.shape)
    rank = max(1, int(min_dim * rank_percentage))
    logging.info(f"Using rank: {rank}")
    
    U = torch.nn.Parameter(torch.randn(original_tensor.shape[0], rank, device=device))
    V = torch.nn.Parameter(torch.randn(rank, original_tensor.shape[1], device=device))
    
    logging.info(f"Created U with shape: {U.shape} and V with shape: {V.shape}")
    
    return U, V

def optimize_low_rank(original_diff, U, V, learning_rate=0.01, num_iterations=1000):
    if original_diff.dim() == 1:
        # For 1D tensors, no optimization is needed
        return 0.0  # Return 0 loss as it's an exact representation
    
    optimizer = torch.optim.Adam([U, V], lr=learning_rate)
    
    for _ in range(num_iterations):
        optimizer.zero_grad()
        approximation = torch.mm(U, V)
        loss = torch.nn.functional.mse_loss(approximation, original_diff)
        loss.backward()
        optimizer.step()
    
    final_loss = loss.item()
    logging.info(f"Final Loss: {final_loss}")
    return final_loss

def process_diff_files(diff_dir, output_dir, rank_percentage=0.1, learning_rate=0.01, num_iterations=1200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ensure_dir(output_dir)
    diff_mapping = {}

    for filename in tqdm(os.listdir(diff_dir), desc="Processing diff files"):
        if filename.endswith('.pt'):
            file_path = os.path.join(diff_dir, filename)
            diff_tensor = torch.load(file_path, map_location=device)
            
            U, V = create_low_rank_tensor(diff_tensor, rank_percentage, device)
            
            final_error = optimize_low_rank(diff_tensor, U, V, learning_rate, num_iterations)
            
            logging.info(f"File: {filename}, Final Error: {final_error}")

            # Save low-rank tensors
            base_name = os.path.splitext(filename)[0]
            output_file = f"{base_name}_lowrank.pt"
            output_path = os.path.join(output_dir, output_file)
            torch.save([U, V], output_path)
            
            # Add to diff_mapping
            original_name = base_name.replace('_', '.')
            diff_mapping[original_name] = output_file

    # Save the diff_mapping
    with open(os.path.join(output_dir, "diff_mapping.json"), 'w') as f:
        json.dump(diff_mapping, f, indent=2)
    logging.info("Created diff_mapping.json in the output directory.")

def main():
    diff_dir = "model_diff"
    output_dir = "optimized_diff"  # New output directory
    rank_percentage = 0.01  # You can adjust this
    learning_rate = 0.008   # You can adjust this
    num_iterations = 5000  # You can adjust this
    
    process_diff_files(diff_dir, output_dir, rank_percentage, learning_rate, num_iterations)

if __name__ == "__main__":
    main()