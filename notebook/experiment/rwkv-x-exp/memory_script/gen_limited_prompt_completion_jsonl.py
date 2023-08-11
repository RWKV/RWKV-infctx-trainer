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
    prompt_templates = [
        "Telephone:\n```\n{document}\n```\n\nGame:",
        "Input:\n```\n{document}\n```\n\nOutput:",
        "Memorise and reply back with the following document:\n```\n{document}\n```\n\nReply:",
        "Document:\n```\n{document}\n```\n\nReply back with the previous document\n\nReply:",
        "Instruction: Repeat this text exactly as it is\n\nInput:\n```\n{document}\n```\n\nResponse:",
        # "Memorize the following document:\n```\n{document}\n```\n\nType it out below:",
        # "Simon:\n```\n{document}\n```\n\nSays:",
        # "Memorize the following document:\n```\n{document}\n```\n\nFor the above document, type it out below:",
        # "Document:\n```\n{document}\n```\n\nReply back with the above document\n\nReply:",
    ]
    completion_templates = [
        "\n```\n{document}\n```\n",
        "\n```\n{document}\n```\n",
        "\n```\n{document}\n```\n",
        "\n```\n{document}\n```\n",
        "\n```\n{document}\n```\n",
        # "\n```\n{document}\n```\n",
        # "\n```\n{document}\n```\n",
        # "\n```\n{document}\n```\n",
        # "\n```\n{document}\n```\n",
    ]

    def get_random_prompt_completion_pair():
        document_lst = []
        word_count = 0

        for i in range(0, max_words, 100):
            paragraph_max = min(max_words - word_count, 100)
            paragraph = random.sample(full_words, min(max(1, random.randint(paragraph_max // 2, paragraph_max)), paragraph_max))
            word_count += paragraph_max
            document_lst.append(' '.join(paragraph))
            document_lst.append("\n\n")

        document = ("".join(document_lst)).strip()

        template_index = random.randint(0, len(prompt_templates) - 1)
        prompt = prompt_templates[template_index].format(document=document)
        completion = completion_templates[template_index].format(document=document)
        return {'prompt': prompt, 'completion': completion}

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