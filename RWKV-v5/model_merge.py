import argparse, math, os
import torch.nn as nn
import torch
from src.model import RWKV 

def model_merge(
        baseline_model_path,
        source_model_path,
        output_model_path=None,
        # Safe merging / skip
        skip_if_exists=False,
        safe_merge=True,
        # The merging mode
        merge_mode="overwrite"
    ):
    # Log the parameters
    print(f"---- Merging model ----")
    print(f'Baseline model path: {baseline_model_path}')
    print(f'Source model path: {source_model_path}')
    print(f'Output model path: {output_model_path}')
    print(f'Merge mode: {merge_mode}')
    print(f"---- ----- ----")

    # Use the baseline model path as the default output model path
    if output_model_path is None:
        output_model_path = baseline_model_path

    # Check if the model exists
    if skip_if_exists and os.path.exists(output_model_path):
        print(f"Model exists, skipping model_merge")
        return

    # Enforce safe_merge if skip_if_exists is set
    if skip_if_exists:
        safe_merge = True

    # Ensure the parent dir exists
    parent_dir = os.path.dirname(output_model_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    
    # Load baseline model
    model_weights = torch.load(baseline_model_path, map_location='cpu')

    # Load the source model
    source_weights = torch.load(source_model_path, map_location='cpu')

    # Iterate through each parameter group in the source model
    # and merge it into the baseline model, if it exists
    for n in source_weights:
        # Operations that are handled if params does not exist in baseline model
        if n not in model_weights:
            print(f"Warning: {n} does not exist in baseline model, skipping")

        # Perform the simple overwrite operation
        if merge_mode == "overwrite":
            model_weights[n] = source_weights[n]
        elif merge_mode == "average":
            model_weights[n] = (model_weights[n] / 2 + source_weights[n] / 2)
        else:
            raise Exception(f"Unknown merge mode: {merge_mode}")
        
        # Ensure values are mapped to CPU & bf16
        model_weights[n] = model_weights[n].cpu().bfloat16()

    # Save the merged model
    if safe_merge:
        # Save as tmp file, then move to the output path
        torch.save(model_weights, output_model_path+".tmp")
        os.rename(output_model_path+".tmp", output_model_path)
    else:
        # Save directly
        torch.save(model_weights, output_model_path)

def main():
    parser = argparse.ArgumentParser(description='CLI tool for model merging')

    # Optional args
    parser.add_argument('--skip-if-exists', type=bool, action=argparse.BooleanOptionalAction, default=False, help='Skip the merge if the model already exists, enables --safe-merge if set')
    parser.add_argument('--safe-merge', type=bool, action=argparse.BooleanOptionalAction, default=False, help='Merge in safe mode, where the model is first merged as a tmp file, before overwritting/moving to the output path')
    parser.add_argument('--merge-mode', type=str, default="overwrite", help='The merge mode, either "overwrite" or "average"')

    # Parse the args
    parser.add_argument('baseline_model_path', type=str, help='Baseline model file path')
    parser.add_argument('source_model_path', type=str, help='Source model file path')
    parser.add_argument('output_model_path', type=str, default=None, help='Output model file path')

    # Parse the args
    args = parser.parse_args()

    # Merge the model
    model_merge(
        args.baseline_model_path,
        args.source_model_path,
        args.output_model_path,
        skip_if_exists=args.skip_if_exists,
        safe_merge=args.safe_merge,
        merge_mode=args.merge_mode
    )

if __name__ == "__main__":
    main()