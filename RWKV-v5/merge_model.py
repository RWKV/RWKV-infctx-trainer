import argparse, math, os
import torch.nn as nn
import torch.nn.functional as F
import torch

### ---
# Torch versioning
### ---

# Because we are using APIs avaliable only to pytorch 2.1.0
# We will throw an error if the user is using a lower version
from packaging import version
def is_torch_version_above(required_version):
    torch_version = version.parse(torch.__version__.split('+')[0])
    return torch_version >= version.parse(required_version)

# Torch versioning flags
IS_TORCH_2_1_COMPATIBLE = is_torch_version_above("2.1.0")

# Model load, using mmap if possible (faster start/lower ram overhead)
def torch_load_model(load_model):
    # Check if model exists
    if not os.path.exists(load_model):
        raise FileNotFoundError(f"Model file not found: {load_model}")

    # Load the model weights
    if IS_TORCH_2_1_COMPATIBLE:
        return torch.load(load_model, map_location='cpu', weights_only=True, mmap=True)
    else:
        return torch.load(load_model, map_location='cpu')

# Extract the model sizing, from the model weights
def extract_model_sizing(model_weights):
    # Get the model keys
    model_keys = list(model_weights.keys())

    # Compute layer count
    max_block_id = 0
    for x in model_keys:
        if 'blocks.' in x:
            block_id = int(x.split('.')[1])
            max_block_id = max(max_block_id, block_id)
    n_layer = max_block_id + 1

    # Compute the embedding, and vocab size
    n_embd = model_weights['head.weight'].shape[1]
    vocab_size = model_weights['head.weight'].shape[0]

    # Return the computed values
    return n_layer, n_embd, vocab_size

# Merge the model within the inner core window
# Given the source and target model weights accordingly
def merge_params_core(source_params, target_params, write_mode="overwrite"):
    # Get the shape of the source and target params
    source_shape = source_params.shape
    target_shape = target_params.shape

    # If its more then 1 dimension, this is probably recursive
    if len(source_shape) > 1:
        
        # Check if the first dimension matches
        if source_shape[0] == target_shape[0]:
            # If it matches, then we will iterate the inner dimensions
            for i in range(source_shape[0]):
                # Merge the inner dimensions
                target_params[i] = merge_params_core(source_params[i], target_params[i], write_mode=write_mode)

        # Shape mismatch, we will set the target_params, from the center
        elif source_shape[0] < target_shape[0]:
            # Compute the center of the target model
            target_center = target_shape[0] // 2
            source_center = source_shape[0] // 2

            # Compute the start and end indices
            start_index = target_center - source_center
            end_index = start_index + source_shape[0]

            # Write the source model to the target model
            for i in range(start_index, end_index):
                target_params[i] = merge_params_core(source_params[i - start_index], target_params[i], write_mode=write_mode)

        # Source is larger then target, raise
        else:
            raise ValueError(f"Source model has larger shape than target model: {source_shape} > {target_shape}")
    
    # If its 1 dimension, then we will perform the write operations (overwrite/average)
    else:

        # Check if the first dimension matches
        if source_shape[0] == target_shape[0]:
            # If it matches, then we will perform the write operations (overwrite/average)
            if write_mode == "overwrite":
                target_params = source_params
            elif write_mode == "average":
                target_params = (target_params + source_params) / 2
            else:
                raise ValueError(f"Invalid write mode: {write_mode}")
        
        # Shape mismatch, we will set the target_params, from the center
        elif source_shape[0] < target_shape[0]:
            # Compute the center of the target model
            target_center = target_shape[0] // 2
            source_center = source_shape[0] // 2

            # Compute the start and end indices
            start_index = target_center - source_center
            end_index = start_index + source_shape[0]

            # Write the source model to the target model
            target_params[start_index:end_index] = source_params
        
        # Source is larger then target, raise
        else:
            raise ValueError(f"Source model has larger shape than target model: {source_shape} > {target_shape}")
    
    # Return the updated target params
    return target_params
    


### ---
# Model merging
### ---
def merge_model(
        source_model_path, target_model_path, output_model_path,
        write_mode="overwrite", resize_mode="core", 
        skip_if_exists=False, safe_init=False
    ):
    
    # ----------------
    # Initial logging
    # ----------------

    # Logging the merge request
    print(f"---- Merging models ----")
    print(f'Source model path: {source_model_path}')
    print(f'Target model path: {target_model_path}')
    print(f'Output model path: {output_model_path}')
    print("")
    print(f'Write mode : {write_mode}')
    print(f'Resize mode: {resize_mode}')
    print(f'Note: this process takes a significant time (and ram) for large models')
    print(f"---- ----- ----")

    # ----------------
    # Output skip check
    # ----------------

    # Skip if exists check
    if skip_if_exists and os.path.exists(output_model_path):
        print(f"Output model exists, skipping init_model")
        return

    # Enforce safe_init if skip_if_exists is set
    if skip_if_exists:
        safe_init = True

    # Ensure the parent dir exists
    parent_dir = os.path.dirname(output_model_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    
    # ----------------
    # Load the models, do basic checks
    # ----------------

    # Load the models
    source_model = torch_load_model(source_model_path)
    target_model = torch_load_model(target_model_path)

    # Extract the model sizing
    source_n_layer, source_n_embd, source_vocab_size = extract_model_sizing(source_model)
    target_n_layer, target_n_embd, target_vocab_size = extract_model_sizing(target_model)

    # Ensure source layer count is less or equal to target layer count
    if source_n_layer > target_n_layer:
        raise ValueError(f"Source model has more layers than target model: {source_n_layer} > {target_n_layer}")
    
    # Ensure source embedding size is less or equal to target embedding size
    if source_n_embd > target_n_embd:
        raise ValueError(f"Source model has larger embedding size than target model: {source_n_embd} > {target_n_embd}")

    # Ensure source vocab size is equal to target vocab size
    if source_vocab_size != target_vocab_size:
        raise ValueError(f"Source model has different vocab size than target model: {source_vocab_size} != {target_vocab_size}")

    # Check the merge and write mode
    write_mode = write_mode.lower()
    if write_mode not in ["overwrite", "average"]:
        raise ValueError(f"Invalid write mode: {write_mode}")
    
    resize_mode = resize_mode.lower()
    if resize_mode not in ["core", "reshape"]:
        raise ValueError(f"Invalid resize mode: {resize_mode}")

    # ----------------

    # Iterate the source model
    for n in source_model:

        # Check if the key exists in the target model, skip if not
        if n not in target_model:
            continue

        # Get source and target param values
        source_param = source_model[n]
        target_param = target_model[n]

        # If the size matches, then we will use it directly from the source model to the target model
        if source_param.shape == target_param.shape:
            print(f"Merge [fit]: {n} - {source_param.shape} -> {target_param.shape}")
            
            if write_mode == "overwrite":
                target_model[n] = source_param
            elif write_mode == "average":
                target_model[n] = (target_param + source_param) / 2

            continue

        # Param sizing check, up to 3 dimensions
        if len(source_param.shape) > 3:
            print(f"SKIP Merge [out-of-bound-shape]: {n} - {source_param.shape} -> {target_param.shape}")
            continue

        # Param sizing check, shape length
        if len(source_param.shape) != len(target_param.shape):
            print(f"SKIP Merge [mismatch-shape-dimensions]: {n} - {source_param.shape} -> {target_param.shape}")
            continue

        # If its reshape mode, then we will reshape the source model to the target model shape
        # And perform the write operations (overwrite/average)
        if resize_mode == "reshape":
            print(f"Merge [reshape]: {n} - {source_param.shape} -> {target_param.shape}")

            interpolate_mode = "linear"

            interploate_param = None
            interpolate_shape = None

            if len(source_param.shape) == 1:
                interpolate_mode = "linear"
                interploate_param = source_param.unsqueeze(0).unsqueeze(0)
                interpolate_shape = [target_param.shape[0]]
            elif len(source_param.shape) == 2:
                interpolate_mode = "bilinear"
                interploate_param = source_param.unsqueeze(0).unsqueeze(0)
                interpolate_shape = [target_param.shape[0], target_param.shape[1]]
            elif len(source_param.shape) == 3:
                interpolate_mode = "trilinear"
                interploate_param = source_param.unsqueeze(0).unsqueeze(0)
                interpolate_shape = [target_param.shape[0], target_param.shape[1], target_param.shape[2]]

            interploate_param = F.interpolate(interploate_param, size=interpolate_shape, mode=interpolate_mode, align_corners=True)
            interploate_param = interploate_param.view(target_param.shape)
            
            if write_mode == "overwrite":
                target_model[n] = interploate_param
            elif write_mode == "average":
                target_model[n] = (target_param + interploate_param) / 2
            continue

        # If its core mode, then we will overwrite the center of the target model with the source model
        # Note this can be multidimensional, so we will need to compute the center
        if resize_mode == "core":
            print(f"Merge [core]: {n} - {source_param.shape} -> {target_param.shape}")
            target_model[n] = merge_params_core(source_param, target_param, write_mode=write_mode)
            continue

        print(f"SKIP Merge [unknown]: {n} - {source_param.shape} -> {target_param.shape}")

    # ----------------

    # Save the model
    if safe_init:
        # Save as tmp file, then move to the output path
        torch.save(target_model, output_model_path+".tmp")
        os.rename(output_model_path+".tmp", output_model_path)
    else:
        # Save directly
        torch.save(target_model, output_model_path)

def main():
    parser = argparse.ArgumentParser(description='CLI tool for model merging')
    parser.add_argument('--skip-if-exists', type=bool, action=argparse.BooleanOptionalAction, default=False, help='Skip the init if the model already exists, enables --safe-init if set')
    parser.add_argument('--safe-init', type=bool, action=argparse.BooleanOptionalAction, default=False, help='Init in safe mode, where the model is first init as a tmp file, before overwritting/moving to the output path')

    parser.add_argument("source_model_path", type=str, help="Path to the source model file (write from)")
    parser.add_argument("target_model_path", type=str, help="Path to the target model file (write to)")
    parser.add_argument('output_model_path', type=str, help='Output model file path')

    parser.add_argument("--write_mode", type=str, default="overwrite", help="`overwrite` / `average` the values from source to target")
    parser.add_argument("--resize_mode", type=str, default="core", help="`core` / `reshape` - core, will overwrite the center of the target model with the source model, reshape will reshape the source model to the target model shape")

    # Parse the args
    args = parser.parse_args()

    merge_model(
        args.source_model_path, args.target_model_path, args.output_model_path, 
        write_mode=args.write_mode, resize_mode=args.resize_mode, 
        skip_if_exists=args.skip_if_exists, safe_init=args.safe_init
    ) #, args.existing_model_path

if __name__ == "__main__":
    main()