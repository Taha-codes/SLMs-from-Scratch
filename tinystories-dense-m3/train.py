from lightning.pytorch.callbacks import TQDMProgressBar
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import lightning as L
from model import GPT, LitGPT
class TinyStoriesDataset(Dataset):
    def __init__(self, data_tokens, seq_len):
        self.data = data_tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len 

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def prepare_dataloader(split="train", num_stories=50000, seq_len=256, batch_size=32, shuffle=True):
    print(f"Loading tokenizer and dataset for {split}...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    hf_dataset = load_dataset("roneneldan/TinyStories", split=split)
    
    small_text_dataset = hf_dataset[:num_stories]['text']
    
    print(f"Tokenizing {num_stories} stories...")
    full_text = " ".join(small_text_dataset)
    
    all_tokens = tokenizer.encode(full_text)
    
    print(f"Total {split} tokens loaded: {len(all_tokens)}")

    dataset = TinyStoriesDataset(all_tokens, seq_len)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader

if __name__ == "__main__":
    SEQ_LEN = 256
    BATCH_SIZE = 64 
    VOCAB_SIZE = 50257
    
    train_loader = prepare_dataloader(split="train", num_stories=20000, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = prepare_dataloader(split="validation", num_stories=2000, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    print("Initializing Model...")
    raw_model = GPT(
        vocab_size=VOCAB_SIZE, 
        d_model=256, 
        n_heads=8, 
        d_ff=1024, 
        seq_len=SEQ_LEN, 
        num_layers=4,    
        dropout=0.1
    )
    
    lit_model = LitGPT(raw_model)
    
    from lightning.pytorch.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='tinystories-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,            
        monitor='val_loss',      
        mode='min'               
    )
    
    print("Starting Training!")
    trainer = L.Trainer(
        max_epochs=1, 
        accelerator="auto", 
        callbacks=[TQDMProgressBar(refresh_rate=100), checkpoint_callback] 
    )
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
