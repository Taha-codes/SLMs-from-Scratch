# TinyStories Small Language Model (SLM)

This project is a PyTorch-based, Transformer architecture Language Model built completely from scratch. It is designed to train on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset and features a modular layout using PyTorch Lightning.

## Architecture

The model is a standard GPT-style autoregressive decoder network consisting of the following key components:

1. **Token Embeddings:** Maps token IDs into continuous vectors (size `d_model`).
2. **Positional Encoding:** Injects sequence order information utilizing the classic sine and cosine formula from *Attention Is All You Need*.
3. **Decoder Blocks:** Includes multiple layers of:
    - **Multi-Head Masked Self-Attention:** Allows tokens to attend strictly to previous context tokens, preventing "look-ahead" cheating via a causal mask.
    - **Feed-Forward Networks (FFN):** A two-layer MLP with a GELU activation wrapper.
    - **Layer Normalization & Residual Connections:** Keeps backpropagation stable across deep layers (Pre-Norm architecture).
4. **Language Model Head:** A linear projection returning the logits distribution over the entire vocabulary.

## Directory Structure

- `model.py`: Contains the core Transformer logic (`GPT`, `DecoderBlock`, `MultiHeadAttention`, etc.) as well as the PyTorch Lightning wrapper class (`LitGPT`).
- `train.py`: The entry point script handling the HuggingFace `TinyStories` download, tokenization (via Hugging Face `gpt2` Byte-Pair Encoding), custom `Dataset`/`DataLoader` formation, and the PyTorch Lightning `Trainer` execution.

## Getting Started

### 1. Install Dependencies
```bash
pip install torch transformers lightning datasets
```

### 2. Begin Training
Run the core script to dynamically download the `TinyStories` train split, encode the text, and spin up an epoch of training:
```bash
python3 train.py
```

### Hyperparameters (Configurable in `train.py`)
- `SEQ_LEN`: 256 (Context window)
- `BATCH_SIZE`: 16 
- `VOCAB_SIZE`: 50257 (GPT-2 standards)
- `d_model`: 256 
- `d_ff`: 1024 
- `n_heads`: 8 
- `num_layers`: 4 
- `dropout`: 0.1 

## Future Scope
- Test generation capabilities by constructing an autoregressive `generate()` loop.
- Scale the architecture size for extended story generation logic.
