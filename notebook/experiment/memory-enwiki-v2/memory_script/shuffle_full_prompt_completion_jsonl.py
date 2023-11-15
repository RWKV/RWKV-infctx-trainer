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
full_words = load_words(os.path.join(os.path.dirname(__file__), 'full_word_list.txt'))

def generate_jsonl(output_file_path, max_words, num_copies):
    prompt_templates = [
        "Telephone:\n```\n{document}\n```\n\nGame:",
        "Input:\n```\n{document}\n```\n\nOutput:",
        "Memorise and reply back with the following document:\n```\n{document}\n```\n\nReply:",
        "Document:\n```\n{document}\n```\n\nReply back with the previous document\n\nReply:",
        "Instruction: Repeat this text exactly as it is\n\nInput:\n```\n{document}\n```\n\nResponse:",
    ]

    completion_templates = [
        "\n```\n{document}\n```\n",
        "\n```\n{document}\n```\n",
        "\n```\n{document}\n```\n",
        "\n```\n{document}\n```\n",
        "\n```\n{document}\n```\n",
    ]

    # Sample counting
    sample_count = 0

    # Write to the output file
    with open(output_file_path, 'w') as f:
        for copy_num in range(num_copies):

            random.shuffle(full_words) 
            full_word_count = len(full_words)
            full_word_used = 0

            def generate_samples(total_used):
                document_lst = []
                word_count = 0

                for i in range(0, max_words, 100):
                    # Get the next paragraph worth of word count
                    paragraph_max = min(max_words - word_count, 100)
                    if paragraph_max <= 0:
                        break
                    paragraph_max = min(max(1, random.randint(paragraph_max // 2, paragraph_max)), paragraph_max)

                    # Get the paragraph itself
                    paragraph = full_words[word_count: word_count + paragraph_max]
                    paragraph_max = len(paragraph)
                    if paragraph_max <= 0:
                        break

                    word_count += paragraph_max
                    total_used += paragraph_max

                    # Append the paragraph to the document
                    document_lst.append(' '.join(paragraph))
                    document_lst.append("\n\n")

                    if (total_used >= full_word_count):
                        break

                document = ("".join(document_lst)).strip()
                template_index = random.randint(0, len(prompt_templates) - 1)
                prompt = prompt_templates[template_index].format(document=document)
                completion = completion_templates[template_index].format(document=document)
                sample = {'prompt': prompt, 'completion': completion}
                f.write(json.dumps(sample) + '\n')

                return total_used

            while full_word_used < full_word_count:
                full_word_used = generate_samples(full_word_used)
                sample_count += 1

    # Return the sample count
    return sample_count

if len(sys.argv) != 4:
    print('Usage: python script.py <outputFilePath> <maxWords> <numCopies>')
    sys.exit(1)

output_file_path, max_words, num_copies = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sample_total = generate_jsonl(output_file_path, max_words, num_copies)

print(f'Generated a single JSONL file with {sample_total} samples ({num_copies} token repeat) - {max_words} max words - at {output_file_path}')