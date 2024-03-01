import json
import tqdm
import argparse

import tokenizers
import numpy as np
from datasets import Dataset


parser = argparse.ArgumentParser(
    description="process preprocessed binaries into dataset for pretraining"
)

parser.add_argument("-o", "--output", required=True, help="output file")
parser.add_argument("-t", "--tokenizer", required=True, help="trained tokenizer file")
parser.add_argument("parsed", nargs="+", help="preprocessed binaries")

arguments = parser.parse_args()

sequence_length = 512
tokenizer = tokenizers.Tokenizer.from_file(arguments.tokenizer)
tokenizer.enable_padding(length=sequence_length)
tokenizer.enable_truncation(max_length=sequence_length)


samples = []
for name in tqdm.tqdm(arguments.parsed, desc="loading"):
    with open(name, "r") as f:
        for line in f:
            function = (json.loads(line))
            function["pretokens"] = " ".join(function["pretokens"])
            samples.append(function)

#samples = {samples}

def process_pretokens(function):
    function["pretokens"] = " ".join(function["pretokens"])
    #pad position_ids to max sequence length. will have to address truncation later
    #function["position_ids"] = np.asarray(function["position_ids"] + )
    #function["position_ids"] = np.asarray(function["position_ids"] + [np.zeros(len(function["position_ids"][0])).tolist()]*(sequence_length - len(function["position_ids"])))
    
    return function


#dataset = load_dataset("json", data_files=arguments.parsed)
#dataset = dataset.map(process_pretokens)
dataset = Dataset.from_list(samples)
print(dataset)
dataset = dataset.train_test_split(train_size=0.8, seed=42)
print(dataset)

print("tokenizing samples...")


def tokenize(batch):
    encoded = tokenizer.encode_batch(batch["pretokens"])
    #encoded = tokenizer.encode_batch(batch)

    batch["input_ids"] = [s.ids for s in encoded]
    batch["attention_mask"] = [s.attention_mask for s in encoded]

    for list in batch["position_ids"]: #truncate position_ids
        #print(len(list))
        if len(list) > sequence_length: #truncate
            for i in range(0, len(list) - sequence_length):
                list.pop()
    #for list in batch["position_ids"]: #pad
        elif len(list) < sequence_length: #pad
            if len(list) != 0:
                for i in range(0, sequence_length - len(list)):
                    list.append(list[0])
            else:
                for i in range(0, sequence_length - len(list)):
                    list.append([0, 0, 0, 0])

    return batch


dataset = dataset.map(tokenize, batched=True, remove_columns=["function", "pretokens"], num_proc=8)

dataset.save_to_disk(arguments.output)
