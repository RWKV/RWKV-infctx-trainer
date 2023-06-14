#!/usr/bin/env python3
import sys
import os
import re
from collections import OrderedDict
import torch

# ----
# This script is used to export the checkpoint into RWKV model
#
# This includes the workaround for a known format issue with the default deepspeed checkpoint exporter
# ----

# Check for argument, else throw error
if len(sys.argv) < 2:
    print("No arguments supplied")
    print("Usage: python3 export_checkpoint.py <checkpoint_dir>")
    sys.exit(1)

# Check if the checkpoint directory exists, else throw error (default assertion)
checkpoint_dir = sys.argv[1]
assert os.path.isdir(checkpoint_dir), "Checkpoint directory does not exist"

# Check if there is an existing zero_to_fp32.py file, in the checkpoint dir, else throw error (default assertion)
zero_to_fp32_path = os.path.join(checkpoint_dir, 'zero_to_fp32.py')
assert os.path.exists(zero_to_fp32_path), "zero_to_fp32.py does not exist in the checkpoint directory"

# We are about to do a search and replace in the zero_to_fp32.py file
# For the following multi-line string
ori_str = r'    print\(f"Saving fp32 state dict to \{output_file\}"\)\n    torch.save\(state_dict, output_file\)'
new_str = '    print(f"Saving fp32 state dict to {output_file}")\n    newDictionary=OrderedDict((k[16:] if k.startswith("_forward_module.") else k, v) for k, v in state_dict.items())\n    torch.save(newDictionary, output_file)'

# Read zero_to_fp32.py file
with open(zero_to_fp32_path, 'r') as file:
    content = file.read()

# Search and replace the following line in the zero_to_fp32.py file
print("# Generating rwkv_zero_to_fp32.py file")
updated_content = re.sub(ori_str, new_str, content)

# Write the modified script to a new file
updated_zero_to_fp32_path = os.path.join(checkpoint_dir, "rwkv_zero_to_fp32.py")
with open(updated_zero_to_fp32_path, 'w') as file:
    file.write(updated_content)

# Run the zero_to_fp32.py file
print("# Running rwkv_zero_to_fp32.py file, exporting the RWKV model")
os.chdir(checkpoint_dir)
os.system(f"python3 rwkv_zero_to_fp32.py . rwkv_model.pth")

# Echo the path to the RWKV model
print("# Exported RWKV model")
print(f"# RWKV fp32 model is located at {checkpoint_dir}/rwkv_model.pth")