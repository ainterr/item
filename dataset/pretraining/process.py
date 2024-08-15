import json
import tqdm
import argparse

import tokenizers

import datasets

import pickle
import networkx as nx
import numpy as np


parser = argparse.ArgumentParser(
    description="process preprocessed binaries into dataset for pretraining"
)

parser.add_argument("-o", "--output", required=True, help="output file")
parser.add_argument("-t", "--tokenizer", required=True, help="trained tokenizer file")
parser.add_argument("parsed", nargs="+", help="preprocessed binaries")

arguments = parser.parse_args()

sequence_length = 512
##eigvec_dim = 4
tokenizer = tokenizers.Tokenizer.from_file(arguments.tokenizer)
#tokenizer.enable_padding(length=sequence_length)
#tokenizer.enable_truncation(max_length=sequence_length)

samples = []
for name in tqdm.tqdm(arguments.parsed, desc="loading"):
    with open(name, "rb") as f:
        data = pickle.load(f)
        data = data.values() #function cfgs
        samples.extend(data)

for idx, item in enumerate(samples): #pickle each function cfg
    samples[idx] = pickle.dumps(item, protocol = 5)

samples = {"cfgs": samples}

dataset = datasets.Dataset.from_dict(samples)
dataset = dataset.train_test_split(train_size=0.8, seed=42)

print("tokenizing samples...")


def tokenize(batch):

    #batch["input_ids"] = [s.ids for s in encoded]
    #batch["attention_mask"] = [s.attention_mask for s in encoded]
    
    for graph_obj in batch["cfgs"]: #parse through each function cfg
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
            batch["input_ids"].extend([input_ids])
            batch["attention_mask"].extend([attention_mask])
            ##batch["position_ids"].extend([position_ids])
        except:
            batch["input_ids"] = [input_ids]
            batch["attention_mask"] = [attention_mask]
            ##batch["position_ids"] = [position_ids]

    return batch


dataset = dataset.map(tokenize, batched=True, remove_columns=["cfgs"], num_proc=8)

dataset.save_to_disk(arguments.output)
