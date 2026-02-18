import os
from datasets import load_dataset #from HuggingFace
import tiktoken
import numpy as np
from tqdm import tqdm

#Setting up the paths
dataset_name = "roneneldan/TinyStories" #pointer to the dataset
data_dir = os.path.join(os.path.dirname(__file__), "data") 
"""
__file__ : the full path to the current file
os.path.dirname(__file__) : the directory of the current file
os.path.join() : join the directory of the current file with "data" to create the path to the data directory
os.path.join(os.path.dirname(__file__), "data") : the path to the data directory
"""

os.makedirs(data_dir, exist_ok=True)

#Loading the dataset
print("Loading dataset...")
dataset = load_dataset(dataset_name)

#The tokenizer (gpt-2 BPE)
enc = tiktoken.get_encoding("gpt2")

def process(example):
    ids = enc.encode_ordinary(example["text"])
    ids.append(enc.eot_token) #end of token
    return {"ids":ids, "len":len(ids)}

#Tokenize
tokenized = dataset.map(
    process,
    remove_columns=["text"],
    desc = "Tokenizing stories",
    num_proc=12
)

for split, dset in tokenized.items():
    filename = os.path.join(data_dir, f'{split}.bin')
    # uint16 is perfect: GPT-2 vocab is 50,257 (fits in 0-65535)
    # This saves 50% memory compared to standard integers!
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))
    
    idx = 0
    for example in tqdm(dset, desc=f"Writing {filename}"):
        arr[idx : idx + example['len']] = example['ids']
        idx += example['len']
    arr.flush()

print(f"Success! Data saved in {data_dir}")