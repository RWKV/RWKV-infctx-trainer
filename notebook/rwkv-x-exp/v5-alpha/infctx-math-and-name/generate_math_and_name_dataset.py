import numpy as np
import argparse

prompt_template = f"""
You are an AI assistant who will be given some tasks to complete.
First, you will be given a name to remember. Then, you will have to sum up a series of numbers.
You will then be asked to answer some questions about the document.

Example 1:
Name: John
1
-2
3
-4

### Question:
What is the total sum?

### Answer:
-2

### Question:
What is the name given at the start of the document?

### Answer:
John

Now you will be tasked to remember the name and sum up the following series of numbers.

"""

task_templates = [
    "\n### Question:\nWhat is the name given at the start of the document?\n\n### Answer:\n",
    "\n### Question:\nWhat is the sum of the numbers given?\n\n### Answer:\n"
]

completion_templates = [
    "\n{name}\n",
    "\n{sum_of_numbers}\n",
]

def load_names(file_path):
    with open(file_path) as word_file:
        valid_names = list(word_file.read().split())
    return valid_names

names = load_names("infctx-math-and-name/names.txt")

def get_random_prompt_completion_pair(max_numbers):
        document = ""
        numbers = np.random.randint(-200, 200, size=(max_numbers))
        total_sum = np.sum(numbers)
        for number in numbers:
            document += str(number) + "\n"

        template_index = np.random.randint(0, len(task_templates))
        task = task_templates[template_index]
        name = names[np.random.randint(0, len(names))]

        prompt = prompt_template + f"Name: {name}\n" + document + task
        completion = completion_templates[template_index].format(sum_of_numbers=total_sum, name=name)
        return {'prompt': prompt, 'completion': completion}

def generate_jsonl(output_file_path, max_numbers, num_samples):
    with open(output_file_path, 'w') as output_file:
        for _ in range(num_samples):
            pair = get_random_prompt_completion_pair(max_numbers)
            output_file.write(str(pair) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-file", type=str, default="questions.jsonl")
    parser.add_argument("--max-numbers", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=10)
    args = parser.parse_args()
    generate_jsonl(args.out_file, args.max_numbers, args.num_samples)