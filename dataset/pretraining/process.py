import json
import tqdm
import argparse

import tokenizers

import datasets


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
        data = json.load(f)
        data = [" ".join(f) for f in data.values() if f]
        samples.extend(data)

samples = {"text": samples}

dataset = datasets.Dataset.from_dict(samples)
dataset = dataset.train_test_split(train_size=0.8, seed=42)

print("tokenizing samples...")


def tokenize(batch):
    encoded = tokenizer.encode_batch(batch["text"])

    batch["input_ids"] = [s.ids for s in encoded]
    batch["attention_mask"] = [s.attention_mask for s in encoded]

    return batch


dataset = dataset.map(tokenize, batched=True, remove_columns=["text"], num_proc=8)

dataset.save_to_disk(arguments.output)
