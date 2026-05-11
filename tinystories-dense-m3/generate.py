import torch
import torch.nn.functional as F
import glob
from transformers import AutoTokenizer
from model import GPT, LitGPT

CHECKPOINT_PATH = "checkpoints/tinystories-epoch=00-val_loss=2.24.ckpt"
DEVICE          = "mps" if torch.backends.mps.is_available() else "cpu"  
MAX_NEW_TOKENS  = 150    
TEMPERATURE     = 0.8    

def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
    model.eval()
    model.to(DEVICE)

    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -256:]

            logits = model(idx_cond)

            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

    return tokenizer.decode(idx[0].tolist())


if __name__ == "__main__":
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"Loading model from: {CHECKPOINT_PATH}")
    lit_model = LitGPT.load_from_checkpoint(
        CHECKPOINT_PATH,
        gpt_model=GPT(
            vocab_size=50257,
            d_model=256,
            n_heads=8,
            d_ff=1024,
            seq_len=256,
            num_layers=4,
        )
    )
    model = lit_model.model
    print(f"Running on: {DEVICE.upper()}\n")

    prompts = [
        "Once upon a time, there was a little dog named",
        "Lily and Tom went to the park. They saw a big",
        "The dragon was very sad because",
    ]

    for prompt in prompts:
        print("─" * 60)
        print(f"PROMPT: {prompt}")
        print("─" * 60)
        story = generate(model, tokenizer, prompt, MAX_NEW_TOKENS, TEMPERATURE)
        print(story)
        print()
