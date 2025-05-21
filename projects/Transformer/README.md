# 🧠 Transformer Language Model

🚀 A clean and minimal PyTorch implementation of a Transformer-based autoregressive language model for text generation, inspired by the paper _"Attention is All You Need"_.

---

## 📚 Table of Contents

- [🔍 Overview](#-overview)  
- [✨ Features](#-features)  
- [⚙️ Installation](#-installation)  
- [📦 Usage](#-usage)  
- [🧱 Model Architecture](#-model-architecture)  
- [🔧 Configuration](#-configuration)  
- [📊 Results](#-results)  
- [🤝 Contributing](#-contributing)  
- [📝 License](#-license)  

---

## 🔍 Overview

This project implements a Transformer model for character or token-level language modeling using **PyTorch**. It includes:

- BERT tokenizer support via HuggingFace 🤗  
- Transformer architecture from scratch  
- Positional embeddings 📍  
- Autoregressive text generation 💬  

---

## ✨ Features

✅ Multi-head self-attention  
✅ Positional encodings  
✅ Layer normalization & residual connections  
✅ BERT-based tokenization  
✅ Custom training loop  
✅ Autoregressive text sampling  
✅ Model checkpointing

---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/transformer-language-model.git
cd transformer-language-model
pip install -r requirements.txt
```
📦 Usage

🚂 Training

To train the model on your text dataset:

```
python train_model.py
```
🔁 Training will:

    1. Load and tokenize text using BERT tokenizer
    2. Train using AdamW optimizer
    3. Print loss every few steps
    4. Save the model checkpoint at the end

✍️ Generating Text
After training, you can generate new text:
    
    import torch
    from model import Transformer
    from transformers import AutoTokenizer
    from utils import decode, DEVICE, BLOCK_SIZE

# Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = Transformer(...)  # Load your config
    model.load_state_dict(torch.load("checkpoints/model.pt"))
    model.to(DEVICE)
    model.eval()

    prompt = "The future of AI"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    generated_ids = model.generate(idx=input_ids, max_new_tokens=50, block_size=BLOCK_SIZE)
    print(decode(generated_ids[0], tokenizer=tokenizer))

🧱 Model Architecture

    Input → [Token Embedding + Positional Embedding] → 
    → Transformer Blocks (N) → 
    → Linear → Softmax
    Each Transformer Block includes:

    💫 Multi-head Self-Attention

    🔄 Residual Connections

    🧮 Layer Normalization

    💥 Feed Forward Network


📊 Results
Example generation after training:

    Prompt: The future of AI
    Output: The future of AI is not only exciting but full of potential and responsibility...
    Loss is printed during training for both training and validation sets.

🤝 Contributing
Contributions are welcome!

🛠️ Feel free to open issues, ask questions, or submit PRs!

📝 License
This project is licensed under the MIT License.
See the LICENSE file for more details.

🙏 Acknowledgements
🔬 "Attention Is All You Need" Paper

🤗 HuggingFace Transformers

Built with ❤️ for learning, building, and sharing.
