import os
import tqdm
import math
import argparse

from sklearn import metrics

import torch
from torch.utils.data import DataLoader

import transformers

import tokenizers

import datasets

import models


parser = argparse.ArgumentParser(
    description="pretrain the model on a masked language modeling dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("-o", "--output", required=True, help="output model directory")
parser.add_argument("-t", "--tokenizer", required=True, help="trained tokenizer file")
parser.add_argument("dataset", help="pretraining dataset directory")

parser.add_argument(
    "-c", "--checkpoint", help="trained model checkpoint from which to resume training"
)
parser.add_argument(
    "-e", "--epochs", type=int, default=10, help="number of epochs for which to train"
)
parser.add_argument("--start-epoch", type=int, default=0, help="starting epoch number")
parser.add_argument("-b", "--batch-size", type=int, default=8, help="batch size")

arguments = parser.parse_args()

os.makedirs(arguments.output, exist_ok=True)

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

model = models.InstructionTraceEncoderTransformerForMaskedLM(configuration)

if arguments.checkpoint:
    state = torch.load(os.path.join(arguments.checkpoint, "pytorch_model.bin"))
    model.load_state_dict(state)

dataset = datasets.load_from_disk(arguments.dataset)

collator = transformers.DataCollatorForLanguageModeling(
    tokenizer=transformers.PreTrainedTokenizerFast(
        tokenizer_file=arguments.tokenizer,
        mask_token="[MASK]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    ),
    mlm_probability=0.15,
)

batch_size = arguments.batch_size
training = DataLoader(
    dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=collator
)
validation = DataLoader(
    dataset["test"], shuffle=True, batch_size=batch_size, collate_fn=collator
)

learning_rate = 1e-4
epochs = arguments.epochs

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

batches = len(training)
steps = epochs * batches
warmup = steps // (2 * epochs)
if arguments.checkpoint:
    warmup = 0

scheduler = transformers.get_scheduler(
    "constant_with_warmup",
    optimizer=optimizer,
    num_warmup_steps=warmup,
    num_training_steps=steps,
)

parallel = False
if torch.cuda.device_count() > 1:
    parallel = True
    model.bert = torch.nn.DataParallel(model.bert)

model.to(models.device)

for epoch in range(arguments.start_epoch, arguments.start_epoch + epochs):
    print(f"epoch {epoch}")

    model.train()
    losses = []
    loop = tqdm.tqdm(training, desc="training")
    for batch in loop:
        batch = {k: v.to(models.device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        loop.set_postfix(loss=sum(losses) / len(losses))

    if parallel:
        model.bert = model.bert.module
        model.save_pretrained(f"{arguments.output}/{epoch}")
        model.bert = torch.nn.DataParallel(model.bert)
    else:
        model.save_pretrained(f"{arguments.output}/{epoch}")

    model.eval()
    values, labels, losses = [], [], []
    performance = {}
    loop = tqdm.tqdm(validation, desc="validating")
    for batch in loop:
        batch = {k: v.to(models.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        references = batch["labels"]
        predictions = torch.argmax(outputs.logits, dim=-1)

        predictions = predictions[references != -100]
        references = references[references != -100]

        values.extend(predictions.tolist())
        labels.extend(references.tolist())
        losses.append(outputs.loss.item())

        # micro-averaged f1 score of masked token prediction
        performance["f1"] = metrics.f1_score(labels, values, average="micro")

        # ppl is ill-defined for masked language modeling, however this is how
        # the code from the Trex paper calcualtes it for masked language
        # pretraining
        loss = sum(losses) / len(losses) / math.sqrt(2)
        performance["ppl"] = 2**loss

        loop.set_postfix(**performance)

    print(f"evaluation performance: {performance}")
