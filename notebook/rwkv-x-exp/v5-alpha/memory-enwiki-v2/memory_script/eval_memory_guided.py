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
    print("Usage: python3 eval_model_memory.py <rwkv_model_path> [verbose/csv-file-path]")
    sys.exit(1)

# Verbose mode
verbose = False
csv_file_path = None
if len(sys.argv) >= 3:
    if sys.argv[2] == "verbose":
        verbose = True
    else:
        csv_file_path = sys.argv[2]

# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

from rwkv.model import RWKV
from rwkv.utils import PIPELINE
from rwkv.utils import PIPELINE_ARGS

# Model strategy to use
# model_run_strat='cpu fp32' # CPU only, use if you dun have a GPU
model_run_strat='cuda fp32' # Entire model is in the GPU (use if you have enough vram)
# model_run_strat='cuda fp32 *11+' # GPU streaming, if you have vram issues for 14B model
# model_run_strat='cuda fp16 *0+' # GPU streaming, if you have really low vram

# Dir of this script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Tokenizer file path
tokenizer_path = os.path.abspath(os.path.join(script_dir,"../../../../RWKV-v4neo/20B_tokenizer.json")) # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV
# Check if file exists
if not os.path.isfile(tokenizer_path):
    print("Tokenizer file not found: ", tokenizer_path)
    sys.exit(1)

# download models: https://huggingface.co/BlinkDL
model_path = sys.argv[1]
model = RWKV(model=model_path, strategy=model_run_strat)
pipeline = PIPELINE(model, tokenizer_path) 

# The evaluation size range
MAX_TOKENS = 1000

# Get the cursed " on" token (this happens only in some models)
on_token = pipeline.encode(" on")[0]
markdown_token = pipeline.encode("```")[0]
newline_token = pipeline.encode("\n")[0]

# Pipeline args to use
token_ban = [on_token] # ban the generation of some tokens
pipeline_args = PIPELINE_ARGS(
                     temperature = 0.2, top_p = 0.2, 
                     top_k = 1, # top_k = 0 then ignore
                     alpha_frequency = 0,
                     alpha_presence = 0,
                     token_ban = token_ban, # ban the generation of some tokens
                     token_stop = [0,markdown_token], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

# Read the test word list, taken from ./eval_word_list.txt
with open(os.path.join(script_dir,'./eval_word_list.txt'), 'r') as f:
    test_word_list = f.read()

# Open the CSV file, to write into
if csv_file_path != None:
    import csv
    csv_file_handle = open(csv_file_path, 'w', newline='')
    csv_writer = csv.writer(csv_file_handle)

    # Write the header
    csv_writer.writerow([
        'eval_token_count', 'token_idx', 'matched', 
        'top_token_str', 'top_token_percentage', 
        'eval_token_str', 'eval_token_pos', 'eval_token_percentage', 
        'is_random_baseline'
    ])
else:
    csv_writer = None

# Convert it to tokens
test_word_tokens = pipeline.encode(test_word_list)

# Prompt template prefix to use
prompt_prefix = "Instruction: Repeat this text exactly as it is\n\nInput:\n```\n"
prompt_suffix = "\n```\n\n"
reply_prefix = "Response:\n```\n"
reply_suffix = "\n```\n"

# Process the prompt prefix
prompt_prefix_logits, prompt_prefix_state = model.forward(pipeline.encode(prompt_prefix), None)
mid_segment_tokens = pipeline.encode(prompt_suffix+reply_prefix)

# Function use to get words with the following token count
def get_words_tokens_with_token_count(token_count):
    target_tokens = test_word_tokens[:token_count]
    target_words = pipeline.decode(target_tokens)
    
    # Normalize to lowercase
    target_words = target_words.lower()
    return target_words

# Function for validating once the model at a specific token count
def validate_model(token_count, withoutInstructAndInput=False):
    # Get the target tokens
    target_tokens = test_word_tokens[:token_count]

    logits = None
    state = None

    # We validate with, the instruct and input
    # having the option to disable this, helps us have a randomized baseline score
    if withoutInstructAndInput == True:
        # Because we actuall need a logit to start with, we compromise with a new line at minimum
        logits, state = model.forward([newline_token], state)
    else:
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
        if verbose or csv_writer != None:
            for j in range(len(sorted_indices)):
                if sorted_indices[j].item() == target:
                    target_pos = j
                    target_prob = sorted_probs[j].item()

            top_token_str = pipeline.decode([top_token])
            target_token_str = pipeline.decode([target])

            # Print the results, for verbose
            if verbose:
                if top_token == target:
                    print(f' - token {i} (hit) : "{top_token_str}" ({top_prob*100:.2f}%)')
                else:
                    print(f' - token {i} (miss): "{top_token_str}" ({top_prob*100:.2f}%) | "{target_token_str}" pos={target_pos} ({target_prob*100:.2f}%)')

            # Log it to CSV file if enabled
            if csv_writer != None:
                # We need to encode the strings safely (escape special characters, new lines, etc)
                top_token_str = top_token_str.encode('unicode_escape').decode('utf-8')
                target_token_str = target_token_str.encode('unicode_escape').decode('utf-8')
                csv_writer.writerow([
                    token_count, i, top_token == target,
                    top_token_str, top_prob,
                    target_token_str, target_pos, target_prob,
                    withoutInstructAndInput == True
                ])
            
        
        # Forward with the target token
        logits, state = model.forward([target], state)
    
    # Percentage token match
    matched_percentage = matched_tokens / token_count * 100.0

    # Print the results
    if withoutInstructAndInput == False:
        print(f'## Model validation for {token_count} tokens : {matched_percentage}% similarity, with {matched_tokens} matched token, and {token_count - matched_tokens} token mismatch')
    else:
        print(f"## Finished baseline model to eval output predictive matching (aka 0 memory?), for {MAX_TOKENS} tokens")
    
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

# We validate in increments of 5, from 5 to 150
for i in range(5, 150, 5):
    validate_model(i)

# We validate in increments of 10 from 150 to 300
for i in range(150, 300, 10):
    validate_model(i)

# We validate in increments of 25 from 300 to 700
for i in range(300, 700, 25):
    validate_model(i)

# We validate in increments of 50 from 700 to 1000
for i in range(700, MAX_TOKENS, 50):
    validate_model(i)

# Lets do the baseline
if csv_file_path != None:
    validate_model(MAX_TOKENS, withoutInstructAndInput=True)

# validate_model(750)
# validate_model(800)
# validate_model(850)
# validate_model(900)
# validate_model(950)
# validate_model(1000)