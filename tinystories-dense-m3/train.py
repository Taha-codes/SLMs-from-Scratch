import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import lightning as L
from model import GPT, LitGPT

# Create a custom PyTorch Dataset
class TinyStoriesDataset(Dataset):
    def __init__(self, data_tokens, seq_len):
        self.data = data_tokens
        self.seq_len = seq_len

    def __len__(self):
        # How many possible sequences can we extract?
        # We subtract seq_len so we don't accidentally ask for tokens past the end!
        return len(self.data) - self.seq_len 

    def __getitem__(self, idx):
        # We grab seq_len tokens for the input
        x = self.data[idx : idx + self.seq_len]
        # We grab the EXACT same sequence, just shifted to the right by 1 for the target
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        
        # PyTorch requires these to be Tensors of type 'long' (integers)
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Setup the DataPipeline
def prepare_dataloader(seq_len=256, batch_size=32):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    print("Downloading TinyStories dataset...")
    # We will just grab the training 'train' split from HuggingFace
    hf_dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    # grab the first 10,000 stories
    small_text_dataset = hf_dataset[:10000]['text']
    
    print("Tokenizing the text...")
    # We convert the list of 10,000 strings into one massive single string
    full_text = " ".join(small_text_dataset)
    
    # Convert the massive string into a massive list of integers
    all_tokens = tokenizer.encode(full_text)
    
    print(f"Total tokens loaded: {len(all_tokens)}")

    # Wrap it in our custom dataset
    dataset = TinyStoriesDataset(all_tokens, seq_len)
    
    # The DataLoader handles shuffling and grouping into batches automatically!
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# 3. The Main Execution
if __name__ == "__main__":
    # Hyperparameters
    SEQ_LEN = 256
    BATCH_SIZE = 16 
    VOCAB_SIZE = 50257 # GPT2 standard vocabulary size
    
    # Prepare data
    train_loader = prepare_dataloader(seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    
    # Initialize Model
    print("Initializing Model...")
    raw_model = GPT(
        vocab_size=VOCAB_SIZE, 
        d_model=256, 
        n_heads=8, 
        d_ff=1024, 
        seq_len=SEQ_LEN, 
        num_layers=4,    # keeping it small so it runs fast!
        dropout=0.1
    )
    
    # Wrap it in PyTorch Lightning
    lit_model = LitGPT(raw_model)
    
    # Train
    print("Starting Training!")
    trainer = L.Trainer(max_epochs=1, accelerator="auto")
    trainer.fit(lit_model, train_dataloaders=train_loader)
