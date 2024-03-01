import os
import re
import json
import logging
import argparse
import collections

import tokenizers
from datasets import load_dataset
import tokens


parser = argparse.ArgumentParser(
    description="train a tokenizer on preprocessed binary data"
)

parser.add_argument("-o", "--output", required=True, help="output file")
parser.add_argument("parsed", nargs="+", help="preprocessed binaries")
parser.add_argument(
    "-v", "--verbose", action="store_true", default=False, help="run in verbose mode"
)

arguments = parser.parse_args()

logging.basicConfig(
    format="%(message)s", level=logging.DEBUG if arguments.verbose else logging.INFO
)

logging.info(f"loading {len(arguments.parsed)} preprocessed binaries...")

dictionary = collections.defaultdict(int)
#print(dictionary)
for p in arguments.parsed:
    functions = []
    #print(p)
    with open(p, "r") as f:
        #file = f"{p}"
        #print(file)
        #functions = load_dataset("json", data_files=file)
        for line in f:
            functions.append(json.loads(line))
        #print(functions)
#functions = json.load(f)
    #print("dataset loaded")

    #for preprocessed in functions["train"]["pretokens"]: #default split is train
    for function in functions:
        for preprocessed in function["pretokens"]:
            for token in preprocessed:
                dictionary[token] += 1
    print("done loading", p)

words, numbers = {}, {}
for token in dictionary.keys():
    try:
        int(token)
        numbers[token] = dictionary[token]
    except:
        words[token] = dictionary[token]

logging.info("done")
logging.info("training tokenizer...")


def dataset(dictionary):
    for token, count in dictionary.items():
        for _ in range(count):
            yield token


tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token=tokens.UNKNOWN))
tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

trainer = tokenizers.trainers.BpeTrainer(
    special_tokens=tokens.ALL, vocab_size=5000, continuing_subword_prefix="__"
)

tokenizer.train_from_iterator(dataset(numbers), trainer=trainer)

tokenizer.add_tokens(list(words))
tokenizer.add_special_tokens(tokens.ALL)

tokenizer.save(arguments.output)

logging.info(f"saved tokenizer to: {arguments.output}")
