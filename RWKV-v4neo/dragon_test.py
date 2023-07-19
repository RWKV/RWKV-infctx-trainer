#!/usr/bin/env python3
import sys
import os

# ----
# This script is used to preload the huggingface dataset
# that is configured in the config.yaml file
# ----

# Check for argument, else throw error
if len(sys.argv) < 2:
    print("No arguments supplied")
    print("Usage: python3 dragon_test.py <model-path> [device]")
    sys.exit(1)

# download models: https://huggingface.co/BlinkDL
MODEL_PATH=sys.argv[1]

# If model strategy is not specified, use 'cpu fp32' as default
DEVICE=None
if len(sys.argv) >= 3:
    DEVICE=sys.argv[2]
if DEVICE is None:
    DEVICE = 'cuda'

# Backward support for older format, we extract only cuda/cpu if its contained in the string
if DEVICE.find('cuda') != -1:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

# Setup the model
from src.model import SimpleRWKV
model = SimpleRWKV(MODEL_PATH, device=DEVICE)

# And perform the dragon prompt
prompt = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
print(f"--- DRAGON PROMPT ---{prompt}", end='')
model.completion(prompt, stream_to_stdout=True, max_tokens=200, temperature=1.0, top_p=0.7)
