#!/usr/bin/env python3
import sys, os
import asyncio

# Special path handling for RWKV code import
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../../../'))
MODEL_CODE_DIR = os.path.join(PROJECT_DIR, 'RWKV-v5headsize32')
sys.path.insert(1, MODEL_CODE_DIR)

#---
# Given the RWKV model path
# Evaluate token memorization capabilities of the model
#
# This uses the model training code instead
#
# Runs on GPU
#---

# Everything in here is done only on the main thread
# this helps resolve wierd issues that occur with doing async / multi-threading
async def main_function():

    # Global last ASYNC anchor operation
    global LAST_ASYNC_OP
    LAST_ASYNC_OP = None

    # Additional import for main function
    import difflib
    import copy
    import torch
    import gc
    from torch.nn import functional as F
    import csv
    import aiofiles
    import time
    from aiocsv import AsyncReader, AsyncDictReader, AsyncWriter, AsyncDictWriter

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
        elif sys.argv[2] == "none":
            csv_file_path = None
        else:
            csv_file_path = sys.argv[2]

    from src.model import SimpleRWKV
    model_path = sys.argv[1]
    model = SimpleRWKV(model_path, device="cuda")

    # The evaluation size range
    MAX_TOKENS = 1000

    # Get the cursed " on" token (this happens only in some models)
    on_token = model.encode(" on")[0]
    markdown_token = model.encode("```")[0]
    newline_token = model.encode("\n")[0]

    # Pipeline args to use
    token_ban = [on_token] # ban the generation of some tokens

    # Read the test word list, taken from ./eval_word_list.txt
    with open(os.path.join(SCRIPT_DIR,'./eval_word_list.txt'), 'r') as f:
        test_word_list = f.read()

    # Open the CSV file, to write into
    if csv_file_path != None:
        # Ensure parent dir is in place
        csv_file_dir = os.path.dirname(csv_file_path)
        if not os.path.exists(csv_file_dir):
            os.makedirs(csv_file_dir)

        # Open the CSV file
        csv_file_handle = await aiofiles.open(csv_file_path, 'w', encoding="utf-8", newline="")
        csv_writer = AsyncWriter(csv_file_handle, dialect="unix")

        # Write the header
        await csv_writer.writerow([
            'eval_token_count', 'token_idx', 'matched', 
            'top_token_str', 'top_token_percentage', 
            'eval_token_str', 'eval_token_pos', 'eval_token_percentage', 
            'is_random_baseline'
        ])
    else:
        csv_writer = None

    # Convert it to tokens
    test_word_tokens = model.encode(test_word_list)

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
    async def validate_model(token_count, withoutInstructAndInput=False):
        # Start the performance timer
        start_time = time.time()
        # print(f"-- Validating model for token count: ", token_count)

        # Get the target tokens
        target_tokens = test_word_tokens[:token_count]

        # Validate that hte token list match the target token count (throw an error if not)
        if len(target_tokens) != token_count:
            raise Exception("Target tokens count mismatch - target is probably larger then the eval word list")

        logits = None
        state = None

        # We validate with, the instruct and input
        # having the option to disable this, helps us have a randomized baseline score
        if withoutInstructAndInput == True:
            # Because we actuall need a logit to start with, we compromise with a new line at minimum
            first_logits, state = model.forward([newline_token], state)
        else:
            # Clone the state
            state = copy.deepcopy(prompt_prefix_state)

            # Compute the document to memorize
            logits, state = model.forward(target_tokens, state)

            # Compute the mid segment
            first_logits, state = model.forward(mid_segment_tokens, state)

        # Score counter
        matched_tokens = 0

        # CSV rows to write
        csv_rows = []

        # Common validation function
        # ----

        async def validateToken(sorted_probs, sorted_indices, softmax_arr, tokenIdx, match_count = 0):
            # Get the top token info
            top_token = sorted_indices[0].item()
            top_prob = sorted_probs[0].item()

            # Check if the token matches, and score it
            target = target_tokens[tokenIdx]
            if top_token == target:
                match_count += 1

            # Find the target token position
            if verbose or csv_writer != None:
                target_prob = softmax_arr[target].item()
                target_pos = 0
                for i in range(len(sorted_indices)):
                    if sorted_indices[i].item() == target:
                        target_pos = i
                        break

                # Get top_token_str & target_token_str, but because an error can happen, we catch it
                try:
                    top_token_str = model.decode([top_token]).encode('unicode_escape').decode('utf-8')
                except:
                    top_token_str = "<UNSAFE_TOKEN_FORMAT>"
                try:
                    target_token_str = model.decode([target]).encode('unicode_escape').decode('utf-8')
                except:
                    target_token_str = "<UNSAFE_TOKEN_FORMAT>"

                # Print the results, for verbose
                if verbose:
                    if top_token == target:
                        print(f' - token {i} (hit) : "{top_token_str}" ({top_prob*100:.2f}%)')
                    else:
                        print(f' - token {i} (miss): "{top_token_str}" ({top_prob*100:.2f}%) | "{target_token_str}" pos={target_pos} ({target_prob*100:.2f}%)')

                # Log it to CSV file if enabled
                if csv_writer != None:
                    # We need to encode the strings safely (escape special characters, new lines, etc)
                    csv_rows.append([
                        token_count, tokenIdx, top_token == target,
                        top_token_str, top_prob,
                        target_token_str, target_pos, target_prob,
                        withoutInstructAndInput == True
                    ])

            # Return matched count
            return match_count
                 
        # Lets validate the first logits
        # ----
   
        # Apply token ban
        for n in token_ban:
            first_logits[n] = -float('inf')

        # Validate the first token (special case)
        first_logits = torch.softmax(first_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(first_logits, descending=True, stable=True, dim=-1)
        matched_tokens = await validateToken(sorted_probs, sorted_indices, first_logits, 0)

        # Print the timing till now
        # print(f"-- Finished validating first token ({time.time() - start_time:.2f}s)")

        # Loop through the target tokens in set of 1000
        # ----
        for subsetPos in range(0, token_count, 1000):

            # Get the subset, and forward it
            token_subset = target_tokens[subsetPos:subsetPos+1000]
            subset_logits, state = model.forward(token_subset, state, all_logits=True)

            # Apply the token ban
            for n in token_ban:
                subset_logits[:,n] = -float('inf')

            # Sort via GPU
            subset_logits = subset_logits.to('cuda')
            subset_logits = torch.softmax(subset_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(subset_logits, descending=True, stable=True, dim=-1)

            # Convert back to CPU land
            sorted_probs = sorted_probs.to('cpu')
            sorted_indices = sorted_indices.to('cpu')

            # Loop through the subset
            for i in range(len(token_subset)):
                pos = i+1+subsetPos
                if pos <= len(target_tokens)-1:
                    matched_tokens = await validateToken(sorted_probs[i], sorted_indices[i], subset_logits[i], pos, matched_tokens)

            # Garbage collect
            gc.collect()
            torch.cuda.empty_cache()
            
        # # Forward all the target tokens in a single pass
        # # ---
        # all_logits, state = model.forward(target_tokens, state, all_logits=True)
        # # print(f"-- Finished multi-token forward pass ({time.time() - start_time:.2f}s)")

        # # Extract the sorted values, and cast them to CPU
        # # ---
        # # Apply token ban
        # for n in token_ban:
        #     all_logits[:,n] = -float('inf')

        # # GPU based sort
        # all_logits = all_logits.to('cuda')
        # all_logits = torch.softmax(all_logits, dim=-1)
        # sorted_probs, sorted_indices = torch.sort(all_logits, descending=True, stable=True, dim=-1)

        # # Convert back to CPU land
        # sorted_probs = sorted_probs.to('cpu')
        # sorted_indices = sorted_indices.to('cpu')

        # # print(f"-- Finished sorting logits ({time.time() - start_time:.2f}s)")

        # # Lets evaluate the logits, and check if they match one by one
        # for i in range(len(target_tokens)-1):
        #     # Validate the token
        #     matched_tokens = await validateToken(sorted_probs[i], sorted_indices[i], all_logits[i], i+1, matched_tokens)

        # print(f"-- Finished token matching ({time.time() - start_time:.2f}s)")

        # Write the CSV rows
        if csv_writer != None:
            await csv_writer.writerows(csv_rows)

        # print(f"-- Finished CSV write ({time.time() - start_time:.2f}s)")
        
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

    # Check if its an extended eval set
    if len(sys.argv) == 4:
        EXTENDED_EVAL = True
        
        # Get the int value from sys.argv[3]
        MAX_TOKENS = int(sys.argv[3])
        MIN_TOKENS = 1100
    elif len(sys.argv) == 5:
        EXTENDED_EVAL = True
        
        # Get the int value from sys.argv[3]/[4]
        MIN_TOKENS = int(sys.argv[3])
        MAX_TOKENS = int(sys.argv[4])
    else:
        EXTENDED_EVAL = False

    # Validate the model at different token counts
    if EXTENDED_EVAL == False:
        # We validate in increments of 5, from 5 to 150
        for i in range(5, 150, 5):
            await validate_model(i)

        # We validate in increments of 10 from 150 to 300
        for i in range(150, 300, 10):
            await validate_model(i)

        # We validate in increments of 25 from 300 to 700
        for i in range(300, 700, 25):
            await validate_model(i)

        # We validate in increments of 50 from 700 to MAXTOKEN (inclusive)
        for i in range(700, MAX_TOKENS+1, 50):
            await validate_model(i)

        # Lets do the baseline
        if csv_file_path != None:
            await validate_model(MAX_TOKENS, withoutInstructAndInput=True)

    else:
        # We validate in increments of 100 from 8000 to MAXTOKEN (inclusive)
        if MAX_TOKENS > 8000:
            for i in range(MIN_TOKENS, MAX_TOKENS+1, 100):
                await validate_model(i)
        else:
            for i in range(MIN_TOKENS, MAX_TOKENS+1, 50):
                await validate_model(i)

    # Print the end of model validation
    print("###")
    print("### Model validation end ###")
    print("###")

if __name__ == '__main__':
    asyncio.run(main_function())