from lightning.pytorch.callbacks import TQDMProgressBar
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
def prepare_dataloader(split="train", num_stories=50000, seq_len=256, batch_size=32, shuffle=True):
    print(f"Loading tokenizer and dataset for {split}...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Grab the requested split from HuggingFace
    hf_dataset = load_dataset("roneneldan/TinyStories", split=split)
    
    # Grab the first N stories
    small_text_dataset = hf_dataset[:num_stories]['text']
    
    print(f"Tokenizing {num_stories} stories...")
    # Convert the list of strings into one massive single string
    full_text = " ".join(small_text_dataset)
    
    # Convert the massive string into a massive list of integers
    all_tokens = tokenizer.encode(full_text)
    
    print(f"Total {split} tokens loaded: {len(all_tokens)}")

    # Wrap it in our custom dataset
    dataset = TinyStoriesDataset(all_tokens, seq_len)
    
    # The DataLoader handles batching automatically!
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader

# 3. The Main Execution
if __name__ == "__main__":
    # Hyperparameters
    SEQ_LEN = 256
    BATCH_SIZE = 128 
    VOCAB_SIZE = 50257 # GPT2 standard vocabulary size
    
    # Prepare data
    # We increased the train stories to 20,000 for a smarter model and lower loss!
    train_loader = prepare_dataloader(split="train", num_stories=20000, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, shuffle=True)
    # We also grab 2,000 stories from the validation split to test the model on unseen data
    val_loader = prepare_dataloader(split="validation", num_stories=2000, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    print("Initializing Model...")
    raw_model = GPT(
        vocab_size=VOCAB_SIZE, 
        d_model=256, 
        n_heads=8, 
        d_ff=1024, 
        seq_len=SEQ_LEN, 
        num_layers=6,    # keeping it small so it runs fast!
        dropout=0.1
    )
    
    # Wrap it in PyTorch Lightning
    lit_model = LitGPT(raw_model)
    
    # Setup Checkpoint Callback to save the model with the lowest validation loss
    from lightning.pytorch.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='tinystories-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,            # Only save the best 1 model
        monitor='val_loss',      # Monitor validation loss
        mode='min'               # We want to minimize the loss
    )
    
    # Train
    print("Starting Training!")
    trainer = L.Trainer(
        max_epochs=1, 
        accelerator="auto", 
        callbacks=[TQDMProgressBar(refresh_rate=100), checkpoint_callback] 
    )
    # We pass both the train AND validation dataloaders now!
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
