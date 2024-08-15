import json
import tqdm
import argparse

import tokenizers

import datasets

import pickle
import networkx as nx
import numpy as np
#from multiprocessing import Pool


parser = argparse.ArgumentParser(
    description="process preprocessed binaries into dataset for pretraining"
)

parser.add_argument("-o", "--output", required=True, help="output file")
parser.add_argument("-t", "--tokenizer", required=True, help="trained tokenizer file")
parser.add_argument("parsed", nargs="+", help="preprocessed binaries")

arguments = parser.parse_args()

sequence_length = 512
tokenizer = tokenizers.Tokenizer.from_file(arguments.tokenizer)
#tokenizer.enable_padding(length=sequence_length)
#tokenizer.enable_truncation(max_length=sequence_length)

samples = []
for name in tqdm.tqdm(arguments.parsed, desc="loading"):
    #with open(name, "r") as f:
        #data = json.load(f)
        #data = [" ".join(f) for f in data.values() if f] #creates one long string
        #samples.extend(data)
    with open(name, "rb") as f:
        data = pickle.load(f)
        data = data.values() #function cfgs
        samples.extend(data)
print("first item of loaded data")
print(samples[0])

for idx, item in enumerate(samples): #pickle each function cfg
    samples[idx] = pickle.dumps(item, protocol = 5)
print("first item")
print(samples[0])

samples = {"cfgs": samples}

dataset = datasets.Dataset.from_dict(samples)
print(dataset)
print(dataset["cfgs"][0])
dataset = dataset.train_test_split(train_size=0.8, seed=42)
print(dataset)

print("tokenizing samples...")


#for identifier, graph in samples["cfgs"].items():

def tokenize(batch):
    x=0
    #encoded = tokenizer.encode_batch(batch["text"])

    #batch["input_ids"] = [s.ids for s in encoded]
    #batch["attention_mask"] = [s.attention_mask for s in encoded]
    
    for graph_obj in batch["cfgs"]: #parse through each function cfg
        #print("start")
        graph = pickle.loads(graph_obj)
        #print("unpickled graph")
        #print(graph)
        laplacian = nx.normalized_laplacian_matrix(graph).toarray()
        eigvals, eigvecs = np.linalg.eig(laplacian)
        idx = eigvals.argsort()
        eigvals, eigvecs = eigvals[idx], np.real(eigvecs[:,idx]) #Sorted by eigenvalue
        #print("eigvecs calculated")
        input_ids = []
        attention_mask = []
        position_ids = []
        for index, node in enumerate(graph.nodes()):
            #print(index)
            #print(node)
            #print(eigvecs[index, 1:]) #ignore trivial eigenvector
            #encoded = tokenizer.encode(node, padding=False, truncation=False)
            node = node.replace(",", "")
            encoded = tokenizer.encode(node)
            if x == 1:
                print(encoded.ids)
            input_ids.extend(encoded.ids)
            #if x == 1:
                #print(input_ids)
            attention_mask.extend(encoded.attention_mask)
            node_lpe = []
            if len(eigvecs[index, 1:]) > 0:
                #if x == 1:
                    #print(eigvecs[index, 1].tolist())
                    #print([eigvecs[index, 1].tolist()])
                for _ in range(len(encoded.ids)):
                    node_lpe.extend([eigvecs[index, 1].tolist()]) #ignore trivial eigenvector and use fielder vector
            else:
                for _ in range(len(encoded.ids)):
                    node_lpe.extend([0])
            position_ids.extend(node_lpe)
            #if x == 1:
                #print(position_ids)
            x=x+1
        #print("lpes calculated")
        if len(input_ids) >= sequence_length:  #truncate
            input_ids = input_ids[:sequence_length]
            attention_mask = attention_mask[:sequence_length]
            position_ids = position_ids[:sequence_length]
        else: #pad
            while len(input_ids) < sequence_length:
                input_ids.append(0)
                attention_mask.append(0)
                position_ids.append(0)
        #print("building dataset")
        try:
            batch["input_ids"].extend([input_ids])
            batch["attention_mask"].extend([attention_mask])
            batch["position_ids"].extend([position_ids])
        except:
            batch["input_ids"] = [input_ids]
            batch["attention_mask"] = [attention_mask]
            batch["position_ids"] = [position_ids]
        #print("next graph")
    
    #batch.pop("cfgs", None)
    #print("about to return")

    return batch


#if __name__ == '__main__':
#pool = Pool(processes=10)                         # Create a multiprocessing Pool
#dataset = pool.apply(tokenize, [samples["cfgs"]])

#samples = tokenize(samples)
#samples = samples.pop(cfgs)
#dataset = datasets.Dataset.from_dict(samples)
#dataset = dataset.train_test_split(train_size=0.8, seed=42)

dataset = dataset.map(tokenize, batched=True, remove_columns=["cfgs"], num_proc=8)
#dataset = dataset.remove_columns("cfgs")

dataset.save_to_disk(arguments.output)


#sequence = sequence[:max_sequence_length]
