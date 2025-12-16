# Disentangled Transformer (RoPE) Implementation

This repository contains a PyTorch implementation of a Transformer language model trained on the Tiny Shakespeare dataset.

## 🚀 Key Architectural Feature: Solving Entanglement
**The Problem:** In standard transformers, adding Word Embeddings + Positional Embeddings creates "entanglement" in the latent space, where word vectors learn positional heuristics and lose semantic purity.
**The Solution:** This implementation uses **Rotary Positional Embeddings (RoPE)**. 
- We **removed** additive positional embeddings from the input.
- We **inject** position information by rotating the Query and Key vectors inside the attention mechanism.
- This ensures the latent space remains cleaner and more interpretable.

## 📂 Project Structure
- `architecture/`: Contains the model definition with RoPE.
- `tokenizer/`: Handles BPE tokenizer training and data splitting.
- `train.py`: Training loop with perplexity logging.
- `test.py`: Inference and test set evaluation.

## 🛠️ Installation & Usage

### 1. Environment Setup
1. First Clone the repository 
```bash
git clone https://github.com/Cheralia/transformer-model.git
```
2. Configure the environment
It is recommended to use a virtual environment.

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt 
```

### 2. Prepare Data & Tokenizer
The tokenizer script will download the dataset, train a BPE tokenizer, and save train.pt, val.pt, and test.pt to the data/ folder.

```bash
# This is called automatically by train.py, but you can run it manually:
python3 tokenizer/tokenizer.py
```
### 3. Training
Run the training loop. This will print the Perplexity (PPL) for every batch.

```bash
python3 train.py
```
### 4. Testing
After training, run evaluation on the held-out test set and generate text:

```bash
python3 test.pyy
```
