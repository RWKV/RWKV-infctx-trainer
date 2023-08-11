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
    print("Usage: python3 dragon_test.py <model-path> [device] [length]") # [tokenizer]")
    sys.exit(1)

# download models: https://huggingface.co/BlinkDL
MODEL_PATH=sys.argv[1]

# If model device is not specified, use 'cuda' as default
DEVICE=None
if len(sys.argv) >= 3:
    DEVICE=sys.argv[2]
IS_REF_RUN = False
if DEVICE == "ref":
    IS_REF_RUN = True

if DEVICE is None:
    DEVICE = 'cuda'

# Get the output length
LENGTH=200
if len(sys.argv) >= 4:
    LENGTH=int(sys.argv[3])

# Backward support for older format, we extract only cuda/cpu if its contained in the string
if DEVICE.find('cuda') != -1:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

# REF run overwrite
if IS_REF_RUN:
    DEVICE = "cpu"

# # Tokenizer settings
# TOKENIZER="neox"
# if len(sys.argv) >= 4:
#     TOKENIZER=sys.argv[3]

# Setup the model
from src.model import SimpleRWKV
model = SimpleRWKV(MODEL_PATH, device=DEVICE)

# Dummy forward, used to trigger any warning / optimizations / etc
model.completion("\n", max_tokens=1, temperature=1.0, top_p=0.7)

# And perform the dragon prompt
prompt = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
if IS_REF_RUN:
    print(f"--- DRAGON PROMPT (REF RUN) ---{prompt}", end='')
    model.completion(prompt, stream_to_stdout=True, max_tokens=LENGTH, temperature=0.0)
else:
    print(f"--- DRAGON PROMPT ---{prompt}", end='')
    model.completion(prompt, stream_to_stdout=True, max_tokens=LENGTH, temperature=1.0, top_p=0.7)

# Empty new line, to make the CLI formatting better
print("")