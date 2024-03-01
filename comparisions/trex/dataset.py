import os
import json
import tqdm
import random
import argparse

import datasets


ARCHITECTURES = ["x86-64", "arm-64", "mips-64"]
OPTIMIZATIONS = ["O0", "O1", "O2", "O3"]

TREX_ARCH_MAPPING = {"x86-64": "x64", "arm-64": "arm", "mips-64": "mips"}


parser = argparse.ArgumentParser(
    description="process preprocessed binaries into dataset for contrastive learning"
)

parser.add_argument("-o", "--output", required=True, help="output file")
parser.add_argument("parsed", nargs="+", help="preprocessed binaries")

arguments = parser.parse_args()

random.seed(42)

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

        data = json.load(f)

        for label, pretokens in data.items():
            label = label.split(":")[-1]

            if label.startswith("0x"):
                # skip unnamed functions
                continue

            name = f"{binary}:{label}"

            if name not in samples:
                samples[name] = {}
            if arch not in samples[name]:
                samples[name][arch] = {}

            samples[name][arch][opt] = pretokens


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

    sample1 = samples[name1][arch1][opt1]
    sample2 = samples[name2][arch2][opt2]

    pairs.append(
        [
            {
                "name": name1,
                "arch": TREX_ARCH_MAPPING[arch1],
                "opt": opt1,
                "static": sample1["static"],
                "instruction": sample1["instruction"],
                "argument": sample1["argument"],
            },
            {
                "name": name2,
                "arch": TREX_ARCH_MAPPING[arch2],
                "opt": opt2,
                "static": sample2["static"],
                "instruction": sample2["instruction"],
                "argument": sample2["argument"],
            },
        ]
    )

    labels.append(matched)

dataset = {
    "arch1": [p[0]["arch"] for p in pairs],
    "static1": [p[0]["static"] for p in pairs],
    "instruction1": [p[0]["instruction"] for p in pairs],
    "argument1": [p[0]["argument"] for p in pairs],
    "arch2": [p[1]["arch"] for p in pairs],
    "static2": [p[0]["static"] for p in pairs],
    "instruction2": [p[0]["instruction"] for p in pairs],
    "argument2": [p[0]["argument"] for p in pairs],
    "label": labels,
}
dataset = datasets.Dataset.from_dict(dataset)

# TODO experiment with unbalanced (representative) classes in test dataset
dataset = dataset.train_test_split(train_size=0.8, seed=42)

dataset.save_to_disk(arguments.output)
