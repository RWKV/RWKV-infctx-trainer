#!/usr/bin/env python3

import sys
import os
import json
import random

def load_words(file_path):
    with open(file_path) as word_file:
        valid_words = list(word_file.read().split())
    return valid_words

# Get the full word list relative to the script
full_words = load_words(os.path.join(os.path.dirname(__file__), 'limited_word_list.txt'))

def generate_jsonl(output_file_path, max_words, num_samples):
    array_template = array_template = [
        {
            'input_prefix': "Input:\n```\n", 
            'output_prefix': "\n```\n\nOutput:\n```\n", 
            'closing':"\n```\n"
        },
        {
            'input_prefix': "Telephone:\n```\n", 
            'output_prefix': "\n```\n\nGame:\n```\n", 
            'closing':"\n```\n"
        },
        {
            'input_prefix': "Memorize the following document:\n```\n", 
            'output_prefix': "\n```\n\nType it out below:\n```\n", 
            'closing':"\n```\n"
        },
        {
            'input_prefix': "Document:\n```\n", 
            'output_prefix': "\n```\n\nReply back with the previous document\n\nReply:\n```\n", 
            'closing':"\n```\n"
        },
        {
            'input_prefix': "Instruction: Repeat this text exactly as it is\n\nInput:\n```\n", 
            'output_prefix': "\n```\n\nResponse:\n```\n", 
            'closing':"\n```\n"
        }
        # {
        #     'input_prefix': "Memorise and reply back with the following document:\n```\n", 
        #     'output_prefix': "```\n\nReply:\n```\n", 
        #     'closing':"\n```\n"
        # },
        # {
        #     'input_prefix': "Simon:\n```\n", 
        #     'output_prefix': "\n```\n\nSays:\n```\n", 
        #     'closing':"\n```\n"
        # },
        # {
        #     'input_prefix': "Memorize the following document:\n```\n", 
        #     'output_prefix': "```\n\nFor the above document, type it out below:\n```\n", 
        #     'closing':"\n```\n"
        # },
        # {
        #     'input_prefix': "Document:\n```\n", 
        #     'output_prefix': "```\n\nReply back with the above document\n\nReply:\n```\n", 
        #     'closing':"\n```\n"
        # },
    ] 

    def get_random_prompt_completion_pair():
        document_lst = []
        word_count = 0

		# Generate random paragraphs, each with 100 words max
		# And merge it into a single document
        for i in range(0, max_words, 100):
            paragraph_max = min(max_words - word_count, 100)
            # This is intentionally biased towards the paragraphMax
            paragraph = random.sample(full_words, min(max(1, random.randint(paragraph_max // 2, paragraph_max)), paragraph_max))
            word_count += paragraph_max
            document_lst.append(' '.join(paragraph))
            document_lst.append("\n\n")

        document = ("".join(document_lst)).strip()
        selected_template = random.choice(array_template)
        return {
            'input_prefix': selected_template['input_prefix'],
            'input': document,
            'output_prefix': selected_template['output_prefix'],
            'output': document,
            'closing': selected_template['closing']
        }

    with open(output_file_path, 'w') as f:
        for i in range(num_samples):
            pair = get_random_prompt_completion_pair()
            f.write(json.dumps(pair) + '\n')


if len(sys.argv) != 4:
    print('Usage: python script.py <outputFilePath> <maxWords> <numSamples>')
    sys.exit(1)

output_file_path, max_words, num_samples = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

generate_jsonl(output_file_path, max_words, num_samples)

print(f'Generated JSONL file with - {max_words} max words, {num_samples} samples - at {output_file_path}')