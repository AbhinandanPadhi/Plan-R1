import os
import torch
from tqdm import tqdm

def verify_dataset(save_root: str, dir_name: str, mode: str, split: str):
    """
    Checks for missing and corrupted preprocessed nuPlan .pt files.
    """
    print(f"\n--- Verifying {dir_name} | {mode} | {split} ---")
    
    # Define paths based on your modified script
    list_file_path = os.path.join(save_root, f"{dir_name}-processed_file_names-{mode}-{split}-PlanR1.pt")
    data_dir = os.path.join(save_root, f"{dir_name}-processed-{mode}-{split}-PlanR1")
    
    if not os.path.exists(list_file_path):
        print(f"Master list file missing: {list_file_path}")
        return
        
    if not os.path.exists(data_dir):
        print(f"Data directory missing: {data_dir}")
        return

    # Load the expected files
    expected_files = torch.load(list_file_path)
    print(f"Found {len(expected_files)} expected files in the index.")
    
    missing_files = []
    corrupted_files = []
    
    for file_name in tqdm(expected_files, desc="Checking files"):
        file_path = os.path.join(data_dir, file_name)
        
        # 1. Check Existence
        if not os.path.exists(file_path):
            missing_files.append(file_name)
            continue
            
        # 2. Check Integrity
        try:
            # Map to CPU to prevent overloading GPU memory during a simple check
            data = torch.load(file_path, map_location='cpu')
            
            # Verify it's a valid dictionary with the expected keys
            if not isinstance(data, dict) or 'scenario_type' not in data:
                corrupted_files.append(file_name)
                
        except Exception:
            corrupted_files.append(file_name)
            
    # Report Results
    if not missing_files and not corrupted_files:
        print(f"Success: All {len(expected_files)} files exist and load perfectly.")
    else:
        if missing_files:
            print(f"Warning: {len(missing_files)} files are missing.")
        if corrupted_files:
            print(f"Warning: {len(corrupted_files)} files are corrupted and cannot be loaded.")

    return missing_files, corrupted_files

if __name__ == '__main__':
    SAVE_ROOT = '/home/jovyan/work/Plan-R1/processed_data'
    
    # Test the configurations you have uncommented in your __main__ block
    # verify_dataset(SAVE_ROOT, dir_name='train_boston', mode='plan', split='train')
    # verify_dataset(SAVE_ROOT, dir_name='train_boston', mode='plan', split='val')
    # verify_dataset(SAVE_ROOT, dir_name='train_boston', mode='pred', split='train')
    # verify_dataset(SAVE_ROOT, dir_name='train_boston', mode='pred', split='val')
    verify_dataset(SAVE_ROOT, dir_name='mini', mode='plan', split='train')
    verify_dataset(SAVE_ROOT, dir_name='mini', mode='plan', split='val')
    verify_dataset(SAVE_ROOT, dir_name='mini', mode='pred', split='train')
    verify_dataset(SAVE_ROOT, dir_name='mini', mode='pred', split='val')