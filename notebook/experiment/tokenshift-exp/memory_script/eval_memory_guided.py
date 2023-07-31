#!/usr/bin/env python3
import sys
import os
import difflib
import copy
import torch
from torch.nn import functional as F

#---
# Given the RWKV model path
# Evaluate token memorization capabilities of the model
#
# Runs on GPU
#---

# Check for argument, else throw error
if len(sys.argv) < 2:
    print("No arguments supplied")
    print("Usage: python3 eval_model_memory.py <rwkv_model_path> [verbose]")
    sys.exit(1)

# Verbose mode
verbose = False
if len(sys.argv) >= 3 and sys.argv[2] == "verbose":
    verbose = True

# Lets load the rwkv v5x model
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../../../'))
V5X_INFER_SRC = os.path.join(PROJECT_DIR, 'RWKV-v5x/')
sys.path.insert(1, V5X_INFER_SRC)
from src.SimpleRWKV import SimpleRWKV

# Device type to use
device_type = 'cuda'

# Dir of this script
model_path = sys.argv[1]

# Setup the SimpleRWKV model
model = SimpleRWKV(model_path, device=device_type)

# Read the test word list, taken from ./eval_word_list.txt
with open(os.path.join(SCRIPT_DIR,'./eval_word_list.txt'), 'r') as f:
    test_word_list = f.read()

# Convert it to tokens
test_word_tokens = model.encode(test_word_list)

# Get the cursed " on" token
on_token = model.encode(" on")[0]
markdown_token = model.encode("```")[0]
# Pipeline args to use
token_ban = [] #[on_token] # ban the generation of some tokens

# Prompt template prefix to use
prompt_prefix = "Instruction: Repeat this text exactly as it is\n\nInput:\n```\n"
prompt_suffix = "\n```\n\n"
reply_prefix = "Response:\n```\n"
reply_suffix = "\n```\n"

# Process the prompt prefix
prompt_prefix_logits, prompt_prefix_state = model.forward(model.encode(prompt_prefix), None)
mid_segment_tokens = model.encode(prompt_suffix+reply_prefix)

# Function use to get words with the following token count
def get_words_tokens_with_token_count(token_count):
    target_tokens = test_word_tokens[:token_count]
    target_words = model.decode(target_tokens)
    
    # Normalize to lowercase
    target_words = target_words.lower()
    return target_words

# Function for validating once the model at a specific token count
def validate_model(token_count):
    # Get the target tokens
    target_tokens = test_word_tokens[:token_count]

    # Clone the state
    state = copy.deepcopy(prompt_prefix_state)

    # Compute the document to memorize
    logits, state = model.forward(target_tokens, state)

    # Compute the mid segment
    logits, state = model.forward(mid_segment_tokens, state)

    # Score counter
    matched_tokens = 0

    # Line break for verbose mode
    if verbose:
        print("## ------------------ ")
        print(f'## Model validation for {token_count} tokens')

    # Lets evaluate the logits, and check if they match one by one
    for i in range(len(target_tokens)):
        # Get the target token
        target = target_tokens[i]

        # Apply token ban
        for n in token_ban:
            logits[n] = -float('inf')

        # We are using a custom sampling method to provide more insight
        # to the probability distribution of the target token

        # Softmax and Sample the logits
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # Get the top token info
        top_token = sorted_indices[0].item()
        top_prob = sorted_probs[0].item()

        # Check if the token matches, and score it
        if top_token == target:
            matched_tokens += 1

        # Find the target token position
        if verbose:
            for j in range(len(sorted_indices)):
                if sorted_indices[j].item() == target:
                    target_pos = j
                    target_prob = sorted_probs[j].item()

            top_token_str = model.decode([top_token])
            target_token_str = model.decode([target])

            # Print the results
            if top_token == target:
                print(f' - token {i} (hit) : "{top_token_str}" ({top_prob*100:.2f}%)')
            else:
                print(f' - token {i} (miss): "{top_token_str}" ({top_prob*100:.2f}%) | "{target_token_str}" pos={target_pos} ({target_prob*100:.2f}%)')
        
        # Forward with the target token
        logits, state = model.forward([target], state)
    
    # Percentage token match
    matched_percentage = matched_tokens / token_count * 100.0

    # Print the results
    print(f'## Model validation for {token_count} tokens : {matched_percentage}% similarity, with {matched_tokens} matched token, and {token_count - matched_tokens} token mismatch')
    if verbose:
        print("## ------------------ ")

    # # Print more info if there are differences
    # if(char_diff_count > 0):
    #     print("---   target   ---")
    #     print(target_words)
    #     print("--- completion ---")
    #     print(completion)
    #     print("------------------")

# Print the start of model validation
print("###")
print("### Model validation start ###")
print("###")

# Validate the model at different token counts
validate_model(5)
validate_model(10)
validate_model(15)
validate_model(20)
validate_model(25)
validate_model(30)
validate_model(35)
validate_model(40)
validate_model(45)
validate_model(50)
validate_model(55)
validate_model(60)
validate_model(65)
validate_model(70)
validate_model(75)
validate_model(80)
validate_model(85)
validate_model(90)
validate_model(95)
validate_model(100)
validate_model(110)
validate_model(120)
validate_model(130)
validate_model(140)
validate_model(150)
validate_model(175)
validate_model(200)
validate_model(225)
validate_model(250)
validate_model(275)
validate_model(300)
validate_model(325)
validate_model(350)
validate_model(375)
validate_model(400)
validate_model(425)
validate_model(450)
validate_model(475)
validate_model(500)
validate_model(550)
validate_model(600)
validate_model(650)
validate_model(700)
# validate_model(750)
# validate_model(800)
# validate_model(850)
# validate_model(900)
# validate_model(950)
# validate_model(1000)