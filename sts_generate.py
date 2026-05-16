from datasets import load_dataset

# 加载GLUE数据集中的STSB子集
dataset = load_dataset("mteb/sts12-sts")

sentence1 = []
sentence2 = []
# 提取相关性高于4的句子对
filtered_data = []
for split in dataset.keys():# 可根据需要选择相应的分割
    for example in dataset[split]:
        if example['score'] > 4: # 如果相关性高于4
            sentence1.append(example['sentence1'])
            sentence2.append(example['sentence2'])

dataset = load_dataset("mteb/sts13-sts")
for split in dataset.keys(): # 可根据需要选择相应的分割
    for example in dataset[split]:
        if example['score'] > 4: # 如果相关性高于4
            sentence1.append(example['sentence1'])
            sentence2.append(example['sentence2'])


dataset = load_dataset("mteb/sts14-sts")
for split in dataset.keys(): # 可根据需要选择相应的分割
    for example in dataset[split]:
        if example['score'] > 4: # 如果相关性高于4
            sentence1.append(example['sentence1'])
            sentence2.append(example['sentence2'])

dataset = load_dataset("mteb/sts15-sts")
for split in dataset.keys(): # 可根据需要选择相应的分割
    for example in dataset[split]:
        if example['score'] > 4: # 如果相关性高于4
            sentence1.append(example['sentence1'])
            sentence2.append(example['sentence2'])

dataset = load_dataset("mteb/sts16-sts")
for split in dataset.keys(): # 可根据需要选择相应的分割
    for example in dataset[split]:
        if example['score'] > 4: # 如果相关性高于4
            sentence1.append(example['sentence1'])
            sentence2.append(example['sentence2'])

dataset = load_dataset("mteb/stsbenchmark-sts")
for split in dataset.keys(): # 可根据需要选择相应的分割
    for example in dataset[split]:
        if example['score'] > 4: # 如果相关性高于4
            sentence1.append(example['sentence1'])
            sentence2.append(example['sentence2'])

file_in = open("sentence1.txt", "w")
for x in sentence1:
    file_in.write(x + "\n")
file_in.close()
file_in = open("sentence2.txt", "w")
for x in sentence2:
    file_in.write(x + "\n")
file_in.close()