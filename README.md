# GPT 124M from Scratch
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)

A complete implementation of a GPT-2 style Large Language Model (124M parameters) built from scratch in PyTorch. This project demonstrates the core components of a Transformer-based language model, including Multi-Head Attention, Feed-Forward Networks, and Positional Embeddings, trained on sample text data.

## 🚀 Features
- **124M Parameter Architecture**: Configuration matching the GPT-2 Small model.
- **Custom Tokenizer**: Uses `tiktoken` for efficient BPE tokenization.
- **Training Loop**: Complete training pipeline with validation and checkpointing.
- **Data Pipeline**: Efficient data loading with sliding window context.
- **Text Generation**: Sampling with temperature and top-k filtering.

## 📂 Project Structure
```
llm-scratch/
├── model/                  # Model architecture
│   ├── gpt_model.py       # Main GPT class
│   ├── attention.py       # Multi-Head Attention
│   ├── transformer_block.py
│   └── transformer_layers.py
├── training/               # Training utilities
│   ├── training.py        # Training loop
│   └── utils.py           # Loss calculation & plotting
├── data/                   # Data handling
│   ├── dataset.py         # PyTorch Dataset
│   └── dataloader.py      # DataLoader factory
├── main.py                 # Entry point for training
└── gpt_124m_config.json    # Model configuration
```

## 🛠️ Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install torch tiktoken matplotlib
```

## 🏃 Usage
1. **Prepare your data**: The project defaults to using `the-verdict.txt`. Ensure this file is present or update `main.py` to point to your dataset.
2. **Configure the model**: stored in `gpt_124m_config.json`.
3. **Start training**:
```bash
python main.py
```
The script will train the model, save the best weights to `gpt_124m.pt`, and generate loss plots.

## 📊 Configuration
The `gpt_124m_config.json` file controls the model hyperparameters:
- `vocab_size`: 50257 (GPT-2 tokenizer)
- `context_length`: 256
- `emb_dim`: 768
- `n_layers`: 12
- `n_heads`: 12
- `batch_size`: 2
- `epochs`: 10

---
*Created for educational purposes to understand LLM internals.*
