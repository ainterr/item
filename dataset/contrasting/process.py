import os
import json
import tqdm
import random
import argparse

import tokenizers

import datasets

import pickle
import networkx as nx
import numpy as np

print("Ghidra 4d vector\n")

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
##eigvec_dim = 10
tokenizer = tokenizers.Tokenizer.from_file(arguments.tokenizer)
#tokenizer.enable_padding(length=sequence_length)
#tokenizer.enable_truncation(max_length=sequence_length)

samples = {}
for name in tqdm.tqdm(arguments.parsed, desc="loading"):
    with open(name, "rb") as f:
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
        data = pickle.load(f)

        #for label, tokens in data.items():
        for label, graph in data.items():
            #label = label.split(":")[-1]
            label = label[0]  #from preprocessing label = (function.getName(), hex(function.getEntryPoint().getOffset()),)

            #if label.startswith("0x"):
            if label.startswith("FUN"):
                # skip unnamed functions
                continue

            name = f"{binary}:{label}"

            #sample = " ".join(tokens)
            #sample = f"[CLS] {sample}"
            sample = pickle.dumps(graph)

            if name not in samples:
                samples[name] = {}
            if arch not in samples[name]:
                samples[name][arch] = {}

            samples[name][arch][opt] = sample


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
                "cfg": samples[name1][arch1][opt1],
            },
            {
                "name": name2,
                "arch": arch2,
                "opt": opt2,
                "cfg": samples[name2][arch2][opt2],
            },
        ]
    )

    labels.append(matched)

dataset = {
    # "name1": [p[0]['name'] for p in pairs],
    # "arch1": [p[0]['arch'] for p in pairs],
    # "opt1": [p[0]['opt'] for p in pairs],
    "cfg1": [p[0]["cfg"] for p in pairs],
    # "name2": [p[1]['name'] for p in pairs],
    # "arch2": [p[1]['arch'] for p in pairs],
    # "opt2": [p[1]['opt'] for p in pairs],
    "cfg2": [p[1]["cfg"] for p in pairs],
    "label": labels,
}
dataset = datasets.Dataset.from_dict(dataset)

# TODO experiment with unbalanced (representative) classes in test dataset
dataset = dataset.train_test_split(train_size=0.8, seed=42)

print("tokenizing samples...")


def tokenize(batch):
    #encoded1 = tokenizer.encode_batch(batch["text1"])
    #encoded2 = tokenizer.encode_batch(batch["text2"])

    #batch["input_ids1"] = [s.ids for s in encoded1]
    #batch["attention_mask1"] = [s.attention_mask for s in encoded1]
    #batch["input_ids2"] = [s.ids for s in encoded2]
    #batch["attention_mask2"] = [s.attention_mask for s in encoded2]
    
    
    for graph_obj in batch["cfg1"]: #parse through each function cfg
        graph = pickle.loads(graph_obj)
        ##laplacian = nx.normalized_laplacian_matrix(graph).toarray()
        ##eigvals, eigvecs = np.linalg.eig(laplacian)
        ##idx = eigvals.argsort()
        ##eigvals, eigvecs = eigvals[idx], np.real(eigvecs[:,idx]) #Sorted by eigenvalue
        input_ids = []
        attention_mask = []
        ##position_ids = []
        for index, node in enumerate(graph.nodes()):
            node = node.replace(",", "")
            node = f"[CLS] {node}"
            encoded = tokenizer.encode(node)
            input_ids.extend(encoded.ids)
            attention_mask.extend(encoded.attention_mask)
            ##node_lpe = []
            ##if len(eigvecs[index, 1:]) > 0:
                ##for _ in range(len(encoded.ids)):
                    ##if len(eigvecs[index, 1:].tolist()) < eigvec_dim:
                        ##partial_node_lpe = eigvecs[index, 1:].tolist()
                        ##pad_length = eigvec_dim - len(eigvecs[index, 1:].tolist())
                        ##partial_node_lpe.extend([0] * pad_length) #needs to match the # of vectors used
                        ##node_lpe.extend([partial_node_lpe])
                    ##else:
                        ##node_lpe.extend([eigvecs[index, 1:eigvec_dim+1].tolist()]) #ignore trivial eigenvector and use eigvec_dim
            ##else:
                ##for _ in range(len(encoded.ids)):
                    ##node_lpe.extend([[0] * eigvec_dim]) #needs to match the # of vectors used
            ##position_ids.extend(node_lpe)
        if len(input_ids) >= sequence_length:  #truncate
            input_ids = input_ids[:sequence_length]
            attention_mask = attention_mask[:sequence_length]
            ##position_ids = position_ids[:sequence_length]
        else: #pad
            while len(input_ids) < sequence_length:
                input_ids.append(0)
                attention_mask.append(0)
                ##position_ids.extend([[0] * eigvec_dim]) #needs to match the # of vectors used
        try:
            batch["input_ids1"].extend([input_ids])
            batch["attention_mask1"].extend([attention_mask])
            ##batch["position_ids1"].extend([position_ids])
        except:
            batch["input_ids1"] = [input_ids]
            batch["attention_mask1"] = [attention_mask]
            ##batch["position_ids1"] = [position_ids]
        
    for graph_obj in batch["cfg2"]: #parse through each function cfg
        graph = pickle.loads(graph_obj)
        ##laplacian = nx.normalized_laplacian_matrix(graph).toarray()
        ##eigvals, eigvecs = np.linalg.eig(laplacian)
        ##idx = eigvals.argsort()
        ##eigvals, eigvecs = eigvals[idx], np.real(eigvecs[:,idx]) #Sorted by eigenvalue
        input_ids = []
        attention_mask = []
        ##position_ids = []
        for index, node in enumerate(graph.nodes()):
            node = node.replace(",", "")
            node = f"[CLS] {node}"
            encoded = tokenizer.encode(node)
            input_ids.extend(encoded.ids)
            attention_mask.extend(encoded.attention_mask)
            ##node_lpe = []
            ##if len(eigvecs[index, 1:]) > 0:
                ##for _ in range(len(encoded.ids)):
                    ##if len(eigvecs[index, 1:].tolist()) < eigvec_dim:
                        ##partial_node_lpe = eigvecs[index, 1:].tolist()
                        ##pad_length = eigvec_dim - len(eigvecs[index, 1:].tolist())
                        ##partial_node_lpe.extend([0] * pad_length) #needs to match the # of vectors used
                        ##node_lpe.extend([partial_node_lpe])
                    ##else:
                        ##node_lpe.extend([eigvecs[index, 1:eigvec_dim+1].tolist()]) #ignore trivial eigenvector and use eigvec_dim
            ##else:
                ##for _ in range(len(encoded.ids)):
                    ##node_lpe.extend([[0] * eigvec_dim]) #needs to match the # of vectors used
            ##position_ids.extend(node_lpe)
        if len(input_ids) >= sequence_length:  #truncate
            input_ids = input_ids[:sequence_length]
            attention_mask = attention_mask[:sequence_length]
            ##position_ids = position_ids[:sequence_length]
        else: #pad
            while len(input_ids) < sequence_length:
                input_ids.append(0)
                attention_mask.append(0)
                ##position_ids.extend([[0] * eigvec_dim]) #needs to match the # of vectors used
        try:
            batch["input_ids2"].extend([input_ids])
            batch["attention_mask2"].extend([attention_mask])
            ##batch["position_ids2"].extend([position_ids])
        except:
            batch["input_ids2"] = [input_ids]
            batch["attention_mask2"] = [attention_mask]
            ##batch["position_ids2"] = [position_ids]

    return batch


dataset = dataset.map(
    tokenize, batched=True, remove_columns=["cfg1", "cfg2"], num_proc=8
)

dataset.save_to_disk(arguments.output)
