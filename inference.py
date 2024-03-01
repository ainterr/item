import os
import time
import argparse

import torch

import tokenizers

from model import models

import preprocess


parser = argparse.ArgumentParser(
    description="predict function similarity",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("-t", "--tokenizer", required=True, help="trained tokenizer file")
parser.add_argument("-m", "--model", required=True, help="trained model directory")
parser.add_argument(
    "-v", "--verbose", action="store_true", default=False, help="enable verbose mode"
)
parser.add_argument("first", help="first function (<binary-path>:<name/rva>")
parser.add_argument("second", help="first function (<binary-path>:<name/rva>")

arguments = parser.parse_args()


def parse(argument):
    binary, name = argument.split(":")
    functions = preprocess.parse(binary)

    tokens = None
    for function in functions:
        for label in function.split(":"):
            if label == name:
                tokens = functions[function]

    if tokens is None:
        print(f"{binary}: unknown function {name}")

        options = sorted(list(functions))
        options = ", ".join(options)

        print(f"options: {options}")

        exit(1)

    sequence = " ".join(tokens)

    if arguments.verbose:
        print(f"{binary}:{name}")
        print("\t" + sequence.replace(" [NEXT] ", "\n\t"))

    return sequence


first = parse(arguments.first)
second = parse(arguments.second)

sequence_length = 512
tokenizer = tokenizers.Tokenizer.from_file(arguments.tokenizer)
tokenizer.enable_padding(length=sequence_length)
tokenizer.enable_truncation(max_length=sequence_length)

configuration = models.InstructionTraceConfig(
    vocab_size=tokenizer.get_vocab_size(),
    next_token_id=tokenizer.token_to_id("[NEXT]"),
    max_position_embeddings=sequence_length,
    type_vocab_size=1,
)

state = torch.load(os.path.join(arguments.model, "pytorch_model.bin"))

model = models.InstructionTraceEncoderTransformerForSequenceSimilarity(configuration)
model.load_state_dict(state)
model.to(models.device)
model.eval()

embedding = model.embedding
difference = model.difference


def tokenize(sequence):
    encoded = tokenizer.encode(sequence)

    tokenized = {}
    tokenized["input_ids"] = torch.tensor(encoded.ids).unsqueeze(0).to(models.device)
    tokenized["attention_mask"] = (
        torch.tensor(encoded.attention_mask).unsqueeze(0).to(models.device)
    )

    return tokenized


first = tokenize(first)
second = tokenize(second)

with torch.no_grad():
    first = embedding(
        input_ids=first["input_ids"],
        attention_mask=first["attention_mask"],
    )

    encoded = first.encode()
    print(f"{arguments.first}: {encoded}")

    second = embedding(
        input_ids=second["input_ids"],
        attention_mask=second["attention_mask"],
    )

    encoded = second.encode()
    print(f"{arguments.second}: {encoded}")

    output = difference(first=first.embedded, second=second.embedded)

similarity = output.item()

print(f"similarity: {similarity}")
