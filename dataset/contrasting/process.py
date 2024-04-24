import os
import json
import tqdm
import random
import argparse

import tokenizers

import datasets

print("Original item2.0 minus small and unconnected functions\n")

ARCHITECTURES = ["x86-64", "arm-64", "mips-64", "x86-32"]
OPTIMIZATIONS = ["O0", "O1", "O2", "O3"]


parser = argparse.ArgumentParser(
    description="process preprocessed binaries into dataset for contrastive learning"
)

parser.add_argument("-o", "--output", required=True, help="output file")
parser.add_argument("-t", "--tokenizer", required=True, help="trained tokenizer file")
parser.add_argument("parsed", nargs="+", help="preprocessed binaries")

arguments = parser.parse_args()

random.seed(42)

sequence_length = 512
tokenizer = tokenizers.Tokenizer.from_file(arguments.tokenizer)
tokenizer.enable_padding(length=sequence_length)
tokenizer.enable_truncation(max_length=sequence_length)

samples = {}
for name in tqdm.tqdm(arguments.parsed, desc="loading"):
    with open(name, "r") as f:
        for arch in ARCHITECTURES:
            if arch in name:
                break
        else:
            print(f"error: unable to detect architecture for: {name}")
            exit(1)

        for opt in OPTIMIZATIONS:
            if f"-{opt}" in name:
                break
        else:
            print(f"error: unable to detect optimization for: {name}")
            exit(1)

        binary = os.path.splitext(os.path.basename(name))[0]
        binary = binary.replace(arch, "").replace(opt, "").strip("-")

        #data = json.load(f)
        #data = datasets.load_dataset("json", data_files=arguments.parsed)
        functions = []
        for line in f:
            function = (json.loads(line))
            functions.append(function)

        data = datasets.Dataset.from_list(functions)

        #for label, tokens in data.items():
        for item in data:
            #label = label.split(":")[-1]
            label = item["function"]

            if label.startswith("sub"):
                # skip unnamed functions
                continue
            
            if not item["position_ids"]:
                continue

            name = f"{binary}:{label}"

            tokens = item["pretokens"]

            sample = " ".join(tokens)
            sample = f"[CLS] {sample}"
            position_ids = [item["position_ids"][0]] + item["position_ids"]

            if name not in samples:
                samples[name] = {}
            if arch not in samples[name]:
                samples[name][arch] = {}
            if opt not in samples[name][arch]:
                samples[name][arch][opt] = {}

            samples[name][arch][opt]["tokens"] = sample
            samples[name][arch][opt]["position_ids"] = position_ids


def select(choices):
    arch = random.choice(list(choices.keys()))
    opt = random.choice(list(choices[arch].keys()))

    return arch, opt


pairs, labels = [], []
for name1 in tqdm.tqdm(sorted(samples.keys()), desc="generating"):
    matched = random.randint(0, 1)

    if len(samples[name1].keys()) == 1:
        arch = list(samples[name1].keys())[0]
        if len(samples[name1][arch].keys()) == 1:
            matched = 0

    arch1, opt1 = select(samples[name1])
    name2, arch2, opt2 = name1, arch1, opt1

    if not matched:
        while name1 == name2:
            name2 = random.choice(list(samples.keys()))

        arch2, opt2 = select(samples[name2])
    else:
        while arch1 == arch2 and opt1 == opt2:
            arch2, opt2 = select(samples[name2])

    pairs.append(
        [
            {
                "name": name1,
                "arch": arch1,
                "opt": opt1,
                "text": samples[name1][arch1][opt1]["tokens"],
                "position_ids": samples[name1][arch1][opt1]["position_ids"]
            },
            {
                "name": name2,
                "arch": arch2,
                "opt": opt2,
                "text": samples[name2][arch2][opt2]["tokens"],
                "position_ids": samples[name2][arch2][opt2]["position_ids"]
            },
        ]
    )

    labels.append(matched)

dataset = {
    # "name1": [p[0]['name'] for p in pairs],
    # "arch1": [p[0]['arch'] for p in pairs],
    # "opt1": [p[0]['opt'] for p in pairs],
    "text1": [p[0]["text"] for p in pairs],
    "position_ids1": [p[0]["position_ids"] for p in pairs],
    # "name2": [p[1]['name'] for p in pairs],
    # "arch2": [p[1]['arch'] for p in pairs],
    # "opt2": [p[1]['opt'] for p in pairs],
    "text2": [p[1]["text"] for p in pairs],
    "position_ids2": [p[1]["position_ids"] for p in pairs],
    "label": labels,
}
dataset = datasets.Dataset.from_dict(dataset)

# TODO experiment with unbalanced (representative) classes in test dataset
dataset = dataset.train_test_split(train_size=0.8, seed=42)

print("tokenizing samples...")


def tokenize(batch):
    encoded1 = tokenizer.encode_batch(batch["text1"])
    encoded2 = tokenizer.encode_batch(batch["text2"])

    batch["input_ids1"] = [s.ids for s in encoded1]
    batch["attention_mask1"] = [s.attention_mask for s in encoded1]
    batch["input_ids2"] = [s.ids for s in encoded2]
    batch["attention_mask2"] = [s.attention_mask for s in encoded2]

    for list in batch["position_ids1"]: #truncate position_ids
        #print(len(list))
        if len(list) > sequence_length: #truncate
            for _ in range(0, len(list) - sequence_length):
                list.pop()
    #for list in batch["position_ids"]: #pad
        elif len(list) < sequence_length: #pad
            if len(list) != 0:
                for _ in range(0, sequence_length - len(list)):
                    list.append(list[0])
            else:
                for _ in range(0, sequence_length - len(list)):
                    list.append([0, 0, 0, 0])

    for list in batch["position_ids2"]: #truncate position_ids
        #print(len(list))
        if len(list) > sequence_length: #truncate
            for _ in range(0, len(list) - sequence_length):
                list.pop()
    #for list in batch["position_ids"]: #pad
        elif len(list) < sequence_length: #pad
            if len(list) != 0:
                for _ in range(0, sequence_length - len(list)):
                    list.append(list[0])
            else:
                for _ in range(0, sequence_length - len(list)):
                    list.append([0, 0, 0, 0])

    return batch


dataset = dataset.map(
    tokenize, batched=True, remove_columns=["text1", "text2"], num_proc=4
)

dataset.save_to_disk(arguments.output)
