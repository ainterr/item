import os
import tqdm
import argparse

from sklearn import metrics
from safetensors.torch import load_file

import torch
from torch.utils.data import DataLoader

import transformers

import tokenizers

import datasets

import models

from accelerate import Accelerator
accelerator = Accelerator()


parser = argparse.ArgumentParser(
    description="fine-tune the model on a contrastive learning dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("-o", "--output", required=True, help="output model directory")
parser.add_argument("-t", "--tokenizer", required=True, help="trained tokenizer file")
parser.add_argument("-m", "--model", required=True, help="pretrained model directory")
parser.add_argument("dataset", help="fine-tuning dataset directory")

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
    frozen_encoder=True,
)

model = models.InstructionTraceEncoderTransformerForSequenceSimilarity(configuration)

#state = torch.load(os.path.join(arguments.model, "pytorch_model.bin"))
state = load_file(os.path.join(arguments.model, "model.safetensors"))
model.embedding.load_state_dict(state, strict=False)

#unwrapped_model = accelerator.unwrap_model(model)
#path_to_checkpoint = os.path.join(arguments.model, "model.safetensors")
#unwrapped_model.load_state_dict(torch.load(path_to_checkpoint))
#unwrapped_model.load_state_dict(load_file(path_to_checkpoint))

dataset = datasets.load_from_disk(arguments.dataset)

collator = transformers.DefaultDataCollator()

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

# warmup = batches // 2
# scheduler = transformers.get_scheduler(
#     "constant_with_warmup",
#     optimizer=optimizer,
#     num_warmup_steps=warmup,
#     num_training_steps=steps
# )

scheduler = transformers.get_scheduler(
    "constant",
    optimizer=optimizer,
    num_training_steps=steps,
)

parallel = False
#if torch.cuda.device_count() > 1:
#    parallel = True
#    model.embedding.bert = torch.nn.DataParallel(model.embedding.bert)

model.to(models.device)

model, optimizer, training, validation, scheduler = accelerator.prepare(
    model, optimizer, training, validation, scheduler)

print(f"model: {arguments.output}")
for epoch in range(arguments.start_epoch, arguments.start_epoch + epochs):
    #print(f"epoch {epoch}")
    accelerator.print(f"epoch {epoch}")

    model.train()
    losses = []
    loop = tqdm.tqdm(training, desc="training")
    for batch in loop:
        #batch = {k: v.to(models.device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        #loss.backward()
        accelerator.backward(loss)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        loop.set_postfix(loss=sum(losses) / len(losses))

    if parallel:
        model.embedding.bert = model.embedding.bert.module
        model.save_pretrained(f"{arguments.output}/{epoch}")
        model.embedding.bert = torch.nn.DataParallel(model.embedding.bert)
    else:
        #model.save_pretrained(f"{arguments.output}/{epoch}")
        accelerator.wait_for_everyone()
        #accelerator.save_model(model, f"{arguments.output}/{epoch}")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            f"{arguments.output}/{epoch}",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )

    model.eval()
    values, labels = [], []
    performance = {}
    loop = tqdm.tqdm(validation, desc="validating")
    for batch in loop:
        #batch = {k: v.to(models.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        references = batch["labels"]
        predictions = outputs.logits.squeeze()
        
        predictions = accelerator.gather_for_metrics(predictions)
        references = accelerator.gather_for_metrics(references)

        labels.extend(references.tolist())
        values.extend(predictions.tolist())

        performance["roc-auc"] = metrics.roc_auc_score(labels, values)

        loop.set_postfix(**performance)

    #print(f"evaluation performance: {performance}")
    accelerator.print(f"evaluation performance: {performance}")
