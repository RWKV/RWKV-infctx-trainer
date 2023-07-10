#!/bin/bash

# This script is used to generate the dataset for the project
mkdir -p dataset
rm -rf dataset/*.jsonl

echo "## Generating word reptition dataset ##"

# # Small sample set, for instruct leanring
# node ./gen_dataset_file.js ./dataset/word-2-count-100.jsonl  2  100 &
# node ./gen_dataset_file.js ./dataset/word-5-count-100.jsonl  5  100 &
# node ./gen_dataset_file.js ./dataset/word-10-count-100.jsonl 10 100 &
# node ./gen_dataset_file.js ./dataset/word-20-count-100.jsonl 20 100 &
# node ./gen_dataset_file.js ./dataset/word-40-count-100.jsonl 40 100 &
# node ./gen_dataset_file.js ./dataset/word-80-count-100.jsonl 80 100 &

# # Really silly 1 - 80 words
# node ./gen_dataset_file.js ./dataset/word-2-count-1k.jsonl  2  1000 &
# node ./gen_dataset_file.js ./dataset/word-5-count-1k.jsonl  5  1000 &
# node ./gen_dataset_file.js ./dataset/word-10-count-1k.jsonl 10 1000 &
# node ./gen_dataset_file.js ./dataset/word-20-count-1k.jsonl 20 1000 &
# node ./gen_dataset_file.js ./dataset/word-40-count-1k.jsonl 40 1000 &
# node ./gen_dataset_file.js ./dataset/word-80-count-1k.jsonl 80 1000 &

node ./gen_dataset_file.js ./dataset/word-2-count-10k.jsonl  2  10000 &
node ./gen_dataset_file.js ./dataset/word-5-count-10k.jsonl  5  10000 &
node ./gen_dataset_file.js ./dataset/word-10-count-10k.jsonl 10 10000 &
node ./gen_dataset_file.js ./dataset/word-20-count-10k.jsonl 20 10000 &
node ./gen_dataset_file.js ./dataset/word-40-count-10k.jsonl 40 10000 &
node ./gen_dataset_file.js ./dataset/word-80-count-10k.jsonl 80 10000 &

# # From 100 to 1000 words
node ./gen_dataset_file.js ./dataset/word-100-count-2k.jsonl 100 2000 &
node ./gen_dataset_file.js ./dataset/word-200-count-2k.jsonl 200 2000 &
node ./gen_dataset_file.js ./dataset/word-300-count-2k.jsonl 300 2000 &
node ./gen_dataset_file.js ./dataset/word-400-count-2k.jsonl 400 2000 &
node ./gen_dataset_file.js ./dataset/word-500-count-2k.jsonl 500 2000 &
node ./gen_dataset_file.js ./dataset/word-600-count-2k.jsonl 600 2000 &
node ./gen_dataset_file.js ./dataset/word-700-count-2k.jsonl 700 2000 &
node ./gen_dataset_file.js ./dataset/word-800-count-2k.jsonl 800 2000 &
node ./gen_dataset_file.js ./dataset/word-900-count-2k.jsonl 900 2000 &

# # From 1k, to 2k
# node ./gen_dataset_file.js ./dataset/word-1k-count-5k.jsonl 1000 5000 &
# node ./gen_dataset_file.js ./dataset/word-2k-count-5k.jsonl 2000 5000 &

# # From 1000 to 10000 words
# node ./gen_dataset_file.js ./dataset/word-1k-count-1k.jsonl 1000 1000 &
# node ./gen_dataset_file.js ./dataset/word-2k-count-1k.jsonl 2000 1000 &
# node ./gen_dataset_file.js ./dataset/word-3k-count-1k.jsonl 3000 1000 &
# node ./gen_dataset_file.js ./dataset/word-4k-count-1k.jsonl 4000 1000 &
# node ./gen_dataset_file.js ./dataset/word-5k-count-1k.jsonl 5000 1000 &
# node ./gen_dataset_file.js ./dataset/word-6k-count-1k.jsonl 6000 1000 &
# node ./gen_dataset_file.js ./dataset/word-7k-count-1k.jsonl 7000 1000 &
# node ./gen_dataset_file.js ./dataset/word-8k-count-1k.jsonl 8000 1000 &
# node ./gen_dataset_file.js ./dataset/word-9k-count-1k.jsonl 9000 1000 &
# node ./gen_dataset_file.js ./dataset/word-10k-count-1k.jsonl 10000 1000 &

wait
echo "## Done ##"
