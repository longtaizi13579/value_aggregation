import random

# Open the file and read all lines
with open("/root/mydata/value_aggregation/eval_dataset/sentence3.txt", "r") as file_in:
    lines = file_in.readlines()

# Randomly select 5409 unique lines
sampled_lines = random.sample(lines, 5409)

# Optional: Write the sampled lines to a new file
with open("/root/mydata/value_aggregation/eval_dataset/sampled_sentence3.txt", "w") as file_out:
    file_out.writelines(sampled_lines)

print(f"Randomly selected {len(sampled_lines)} samples and saved to 'sampled_sentence3.txt'.")