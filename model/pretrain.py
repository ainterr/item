import os
import tqdm
import math
import argparse

from sklearn import metrics

import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file

import transformers

import tokenizers

import datasets

import models

from accelerate import Accelerator
accelerator = Accelerator()

#import torch.multiprocessing as mp ##
#from torch.utils.data.distributed import DistributedSampler ##
#from torch.nn.parallel import DistributedDataParallel as DDP ##
#from torch.distributed import init_process_group, destroy_process_group ##

accelerator.print("Ghidra fiedler vector w/o cudann benchmarch \n")
accelerator.print("no loss gather, separate gather_for_metrics")

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

#def ddp_setup(rank: int, world_size: int): ##
#    """
#    Args:
#        rank: Unique identifier of each process
#        world_size: Total number of processes
#    """
#    os.environ["MASTER_ADDR"] = "localhost"
#    os.environ["MASTER_PORT"] = "12355"
#    torch.cuda.set_device(rank)
#    init_process_group(backend="nccl", rank=rank, world_size=world_size)

sequence_length = 512
tokenizer = tokenizers.Tokenizer.from_file(arguments.tokenizer)
#tokenizer.enable_padding(length=sequence_length)
#tokenizer.enable_truncation(max_length=sequence_length)

configuration = models.InstructionTraceConfig(
    vocab_size=tokenizer.get_vocab_size(),
    next_token_id=tokenizer.token_to_id("[NEXT]"),
    max_position_embeddings=sequence_length,
    type_vocab_size=1,
)

model = models.InstructionTraceEncoderTransformerForMaskedLM(configuration)

if arguments.checkpoint:
    #state = torch.load(os.path.join(arguments.checkpoint, "pytorch_model.bin"))
    
    state = load_file(os.path.join(arguments.model, "model.safetensors"))
    model.load_state_dict(state)
    
    #unwrapped_model = accelerator.unwrap_model(model)
    #path_to_checkpoint = os.path.join(arguments.model, "model.safetensors")
    #unwrapped_model.load_state_dict(torch.load(path_to_checkpoint))

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
training = DataLoader(dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=collator)
#training = DataLoader(dataset["train"], shuffle=False, batch_size=batch_size, collate_fn=collator, sampler=DistributedSampler(dataset["train"]))
validation = DataLoader(dataset["test"], shuffle=True, batch_size=batch_size, collate_fn=collator)
#validation = DataLoader(dataset["test"], shuffle=True, batch_size=batch_size, collate_fn=collator, sampler=DistributedSampler(dataset["test"]))

learning_rate = 1e-4
epochs = arguments.epochs

model.to(models.device)
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
#if torch.cuda.device_count() > 1:
    #parallel = True
    #model.bert = torch.nn.DataParallel(model.bert)

#model = DDP(model, device_ids=[gpu_id]) ##

model, optimizer, training, validation, scheduler = accelerator.prepare(
    model, optimizer, training, validation, scheduler
)

#torch.backends.cudnn.benchmark = True
for epoch in range(arguments.start_epoch, arguments.start_epoch + epochs):
    accelerator.print(f"epoch {epoch}")
    #training.sampler.set_epoch(epoch) ##

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
        model.bert = model.bert.module
        model.save_pretrained(f"{arguments.output}/{epoch}")
        model.bert = torch.nn.DataParallel(model.bert)
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
    values, labels, losses = [], [], []
    performance = {}
    loop = tqdm.tqdm(validation, desc="validating")
    for batch in loop:
        #batch = {k: v.to(models.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        references = batch["labels"]
        predictions = torch.argmax(outputs.logits, dim=-1)
        #loss_values = outputs.loss.item()
        
        predictions = accelerator.gather_for_metrics(predictions)
        references = accelerator.gather_for_metrics(references)

        predictions = predictions[references != -100]
        references = references[references != -100]

        values.extend(predictions.tolist())
        labels.extend(references.tolist())
        losses.append(outputs.loss.item())
        
        # micro-averaged f1 score of masked token prediction
        performance["f1"] = metrics.f1_score(labels, values, average="micro")

        # ppl is ill-defined for masked language modeling, however this is how
        # the code from the Trex paper calculates it for masked language
        # pretraining
        loss = sum(losses) / len(losses) / math.sqrt(2)
        performance["ppl"] = 2**loss

        loop.set_postfix(**performance)

    accelerator.print(f"evaluation performance: {performance}")
