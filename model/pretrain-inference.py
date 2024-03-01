import os
import argparse

import torch

import transformers

import models


parser = argparse.ArgumentParser(
    description="predict masked instruction trace tokens",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("-t", "--tokenizer", required=True, help="trained tokenizer file")
parser.add_argument("-m", "--model", required=True, help="trained model directory")
parser.add_argument(
    "sequence", help="sequence of tokenizable inputs, including mask tokens"
)

arguments = parser.parse_args()

sequence_length = 512

tokenizer = transformers.PreTrainedTokenizerFast(
    tokenizer_file=arguments.tokenizer,
    mask_token="[MASK]",
    unk_token="[UNK]",
    pad_token="[PAD]",
)

configuration = models.InstructionTraceConfig(
    vocab_size=len(tokenizer.vocab),
    next_token_id=tokenizer("[NEXT]")["input_ids"][0],
    max_position_embeddings=sequence_length,
    type_vocab_size=1,
)

model = models.InstructionTraceEncoderTransformerForMaskedLM(configuration)
state = torch.load(os.path.join(arguments.model, "pytorch_model.bin"))
model.load_state_dict(state)

model.to(models.device)
model.eval()

tokens = tokenizer(arguments.sequence, return_tensors="pt").to(models.device)

with torch.no_grad():
    output = model(**tokens)

tokens = torch.argmax(output.logits[0], dim=-1)

decoded = tokenizer.decode(tokens)

print(decoded)
