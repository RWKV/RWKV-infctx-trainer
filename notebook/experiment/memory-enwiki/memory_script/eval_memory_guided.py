#!/usr/bin/env python3
import sys
import os
import difflib
import copy

#---
# Given the RWKV model path
# Evaluate token memorization capabilities of the model
#
# Runs on GPU
#---

# Check for argument, else throw error
if len(sys.argv) < 2:
    print("No arguments supplied")
    print("Usage: python3 eval_model_memory.py <rwkv_model_path>")
    sys.exit(1)

# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

from rwkv.model import RWKV
from rwkv.utils import PIPELINE
from rwkv.utils import PIPELINE_ARGS

# Model strategy to use
# model_run_strat='cpu fp32' # CPU only, use if you dun have a GPU
model_run_strat='cuda fp32' # Entire model is in the GPU (use if you have enough vram)
# model_run_strat='cuda fp16 *30+' # GPU streaming, if you have vram issues for 14B model
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

# Get the cursed " on" token
on_token = pipeline.encode(" on")[0]
markdown_token = pipeline.encode("```")[0]

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

    # Lets evaluate the logits, and check if they match one by one
    for i in range(len(target_tokens)):
        # Get the target token
        target = target_tokens[i]

        # Apply token ban
        for n in token_ban:
            logits[n] = -float('inf')

        # Sample the logits
        token = pipeline.sample_logits(logits, 0.1, 0.0, 1)

        # Check if the token matches, and score it
        if token == target:
            matched_tokens += 1

        # Forward with the target token
        logits, state = model.forward([target], state)
    
    # Percentage token match
    matched_percentage = matched_tokens / token_count * 100.0

    # Print the results
    print(f'Model validation at {token_count} tokens : {matched_percentage}% similarity, with {matched_tokens} matched token, and {token_count - matched_tokens} token mismatch')

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