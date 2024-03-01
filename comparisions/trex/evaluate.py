import tqdm
import argparse

import torch
from torch.utils.data import DataLoader

from fairseq.models.trex import TrexModel

import datasets

import transformers

from sklearn import metrics


def preprocess(static, arch, instruction, argument):
    sample = {}

    length = min(len(static), 512)

    sample["static"] = " ".join(static[:length])
    sample["inst_pos_emb"] = " ".join([str(i) for i in instruction[:length]])
    sample["op_pos_emb"] = " ".join([str(i) for i in argument[:length]])
    sample["arch_emb"] = f"{arch} " * length

    byte = "## " * length

    sample["byte1"] = byte
    sample["byte2"] = byte
    sample["byte3"] = byte
    sample["byte4"] = byte

    return sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate TREX model")

    parser.add_argument("-m", "--model", required=True, help="trained model checkpoint")
    parser.add_argument("dataset", help="trex evaluation dataset directory")

    arguments = parser.parse_args()

    model = TrexModel.from_pretrained(
        arguments.model,
        checkpoint_file="checkpoint_best.pt",
        data_name_or_path=arguments.model,
    )
    model = model.cpu()
    model.eval()

    optimized = torch.jit.script(model.model)

    dataset = datasets.load_from_disk(arguments.dataset)

    # def embed(batch):
    #    embedded1, embedded2 = [], []

    #    for i, label in enumerate(batch['label']):
    #        sample1 = preprocess(
    #            static=batch['static1'][i],
    #            arch=batch['arch1'][i],
    #            instruction=batch['instruction1'][i],
    #            argument=batch['argument1'][i]
    #        )
    #        sample2 = preprocess(
    #            static=batch['static2'][i],
    #            arch=batch['arch2'][i],
    #            instruction=batch['instruction2'][i],
    #            argument=batch['argument2'][i]
    #        )

    #        encoded1 = model.encode(sample1)
    #        encoded2 = model.encode(sample2)

    #        processed1 = model.process_token_dict(encoded1)
    #        processed2 = model.process_token_dict(encoded2)

    #        embedded1.append(optimized(processed1, features_only=True, classification_head_name='similarity')[0]['features'])
    #        embedded2.append(optimized(processed2, features_only=True, classification_head_name='similarity')[0]['features'])

    #    batch['embedded1'] = embedded1
    #    batch['embedded2'] = embedded2

    #    return batch

    # dataset = dataset.map(embed, batched=True, remove_columns=["static1", "arch1", "instruction1", "argument1", "static2", "arch2", "instruction2", "argument2"], batch_size=1)

    collator = transformers.DefaultDataCollator()

    validation = DataLoader(
        dataset["test"], shuffle=True, batch_size=64, collate_fn=collator
    )

    values, labels = [], []
    performance = {}
    loop = tqdm.tqdm(validation, desc="validating")
    for batch in loop:
        predictions = torch.cosine_similarity(
            batch["embedded1"], batch["embedded2"], dim=-1
        ).squeeze()

        labels.extend(batch["labels"].tolist())
        values.extend(predictions.tolist())

        performance["roc-auc"] = metrics.roc_auc_score(labels, values)

        loop.set_postfix(**performance)

    print(f"evaludation performance: {performance}")
