#!/bin/bash

# This script is used to generate the dataset for the project
mkdir -p dataset
rm -rf dataset/*.jsonl

echo "## Generating word reptition dataset ##"

# We do a strong bias for smaller word count, so that the model can learn the function
node ./gen_dataset_file.js ./dataset/word-2-count.jsonl  2  20000 &
node ./gen_dataset_file.js ./dataset/word-5-count.jsonl  5  20000 &
node ./gen_dataset_file.js ./dataset/word-10-count.jsonl 10 20000 &
node ./gen_dataset_file.js ./dataset/word-20-count.jsonl 20 20000 &
node ./gen_dataset_file.js ./dataset/word-40-count.jsonl 40 20000 &
node ./gen_dataset_file.js ./dataset/word-80-count.jsonl 80 20000 &

# With a slight mix of the larger word count
node ./gen_dataset_file.js ./dataset/word-100-count.jsonl 100 8000 &
node ./gen_dataset_file.js ./dataset/word-200-count.jsonl 200 5500 &
node ./gen_dataset_file.js ./dataset/word-300-count.jsonl 300 5000 &
node ./gen_dataset_file.js ./dataset/word-400-count.jsonl 400 4500 &
node ./gen_dataset_file.js ./dataset/word-500-count.jsonl 500 4000 &
node ./gen_dataset_file.js ./dataset/word-600-count.jsonl 600 3500 &
node ./gen_dataset_file.js ./dataset/word-700-count.jsonl 700 3000 &
node ./gen_dataset_file.js ./dataset/word-800-count.jsonl 800 2500 &
node ./gen_dataset_file.js ./dataset/word-900-count.jsonl 900 2000 &

wait
echo "## Done ##"
