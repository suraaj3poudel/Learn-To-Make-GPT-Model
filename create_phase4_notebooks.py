#!/usr/bin/env python3
"""
Generate Phase 4 notebooks - Build Your GPT!
Final phase: Complete GPT implementation with chat interface
"""
import json
import os

def save_notebook(filepath, cells):
    """Save properly formatted Jupyter notebook"""
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.6"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(notebook, f, indent=2)
    print(f"✅ Created: {filepath}")

def md(text): return {"cell_type": "markdown", "metadata": {}, "source": text.split('\n')}
def code(text): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.split('\n')}

print("="*60)
print("🎯 Creating Phase 4: Build Your GPT!")
print("="*60)

# ============================================================================
# PHASE 4, LESSON 1: GPT ARCHITECTURE
# ============================================================================
print("\n📝 Creating Phase 4, Lesson 1: GPT Architecture...")

phase4_lesson1 = [
    md("""# Phase 4, Lesson 1: GPT Architecture

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suraaj3poudel/Learn-To-Make-GPT-Model/blob/main/phase4_build_your_gpt/01_gpt_architecture.ipynb)

Build your own GPT from scratch! 🤖

## What You'll Learn
1. GPT architecture (decoder-only transformer)
2. Modern techniques (dropout, weight tying, etc.)
3. Using PyTorch for real implementation
4. Model configuration and scaling

The final boss! Let's do this!"""),

    code("""# Setup with PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ Device: {\"cuda\" if torch.cuda.is_available() else \"cpu\"}')"""),

    md("""## 1. GPT vs. Full Transformer

**Key Differences**:
- GPT = **Decoder-only** (no encoder)
- Uses **causal/masked** attention (can't see future)
- Trained on **autoregressive** language modeling

**Architecture**:
```
Input tokens
    ↓
Token + Position Embeddings
    ↓
Transformer Block 1
    ↓
Transformer Block 2
    ↓
... (many blocks)
    ↓
Transformer Block N
    ↓
Layer Norm
    ↓
Linear (to vocabulary)
    ↓
Output probabilities
```"""),

    md("""## 2. Configuration

Define model hyperparameters"""),

    code("""class GPTConfig:
    \"\"\"Configuration for GPT model\"\"\"
    def __init__(
        self,
        vocab_size=50257,      # GPT-2 vocabulary size
        n_positions=1024,       # Maximum sequence length
        n_embd=768,            # Embedding dimension
        n_layer=12,            # Number of transformer blocks
        n_head=12,             # Number of attention heads
        dropout=0.1,           # Dropout probability
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout

# Example configurations
config_small = GPTConfig(
    vocab_size=10000,
    n_positions=256,
    n_embd=128,
    n_layer=4,
    n_head=4,
)

config_medium = GPTConfig(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
)

print("Small GPT config:")
print(f"  Params: ~{(config_small.n_layer * config_small.n_embd ** 2 * 12) / 1e6:.1f}M")
print(f"  Embedding dim: {config_small.n_embd}")
print(f"  Layers: {config_small.n_layer}")"""),

    md("""## 3. Multi-Head Attention (PyTorch)

Professional implementation with PyTorch"""),

    code("""class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Key, query, value projections for all heads (in one matrix)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # Causal mask (lower triangular)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions))
                  .view(1, 1, config.n_positions, config.n_positions)
        )
    
    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim
        
        # Calculate Q, K, V for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Split into multiple heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        # Attention: (Q @ K^T) / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(k.size(-1))))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

# Test
config = config_small
mha = MultiHeadAttention(config)
x = torch.randn(2, 10, config.n_embd)  # Batch of 2, sequence of 10
output = mha(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print("✅ Multi-head attention working!")"""),

    md("""## 4. Feed-Forward Network

MLP with GELU activation (used in GPT)"""),

    code("""class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# Test
ffn = FeedForward(config)
x = torch.randn(2, 10, config.n_embd)
output = ffn(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")"""),

    md("""## 5. Transformer Block

Combine attention and feed-forward with residuals and layer norm"""),

    code("""class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)
    
    def forward(self, x):
        # Attention with residual
        x = x + self.attn(self.ln_1(x))
        
        # Feed-forward with residual
        x = x + self.mlp(self.ln_2(x))
        
        return x

# Test
block = TransformerBlock(config)
x = torch.randn(2, 10, config.n_embd)
output = block(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print("✅ Transformer block working!")"""),

    md("""## 6. Complete GPT Model

Put it all together!"""),

    code("""class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token + position embeddings
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
            wpe = nn.Embedding(config.n_positions, config.n_embd),  # Position embeddings
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying (share weights between token embedding and output)
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.n_positions, f"Sequence length {t} exceeds maximum {self.config.n_positions}"
        
        # Position indices
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)
        
        # Forward pass
        tok_emb = self.transformer.wte(idx)  # Token embeddings: (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # Position embeddings: (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        
        # Language model head
        logits = self.lm_head(x)  # (b, t, vocab_size)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        \"\"\"
        Generate text autoregressively
        
        Args:
            idx: (b, t) array of indices
            max_new_tokens: How many tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens
        \"\"\"
        for _ in range(max_new_tokens):
            # Crop to max length
            idx_cond = idx if idx.size(1) <= self.config.n_positions else idx[:, -self.config.n_positions:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Get logits for last position
            logits = logits[:, -1, :] / temperature
            
            # Optional: crop to top k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

# Create model
config = config_small
model = GPT(config)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"GPT model created!")
print(f"Parameters: {n_params:,}")
print(f"Model size: ~{n_params/1e6:.2f}M parameters")

# Test forward pass
x = torch.randint(0, config.vocab_size, (2, 10))
logits, loss = model(x, targets=x)

print(f"\\nInput shape: {x.shape}")
print(f"Output logits shape: {logits.shape}")
print(f"\\n✅ Complete GPT model working!")"""),

    md("""## 7. Model Sizes

Compare different GPT scales:"""),

    code("""# Different model configurations
configs = {
    "Tiny": GPTConfig(vocab_size=10000, n_embd=64, n_layer=2, n_head=2, n_positions=128),
    "Small": GPTConfig(vocab_size=10000, n_embd=128, n_layer=4, n_head=4, n_positions=256),
    "Medium": GPTConfig(vocab_size=10000, n_embd=256, n_layer=6, n_head=8, n_positions=512),
    "GPT-2 Small": GPTConfig(vocab_size=50257, n_embd=768, n_layer=12, n_head=12, n_positions=1024),
}

print("Model Size Comparison:\\n")
for name, cfg in configs.items():
    model = GPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name:15} {n_params/1e6:8.2f}M parameters")

print("\\nGPT-3 has 175 BILLION parameters! 🤯")"""),

    md("""## Summary

### What We Built:
1. **GPT Configuration** - Flexible model sizing
2. **Multi-Head Attention** - Professional PyTorch implementation
3. **Transformer Blocks** - With residuals and layer norm
4. **Complete GPT Model** - Ready for training!
5. **Generation Method** - Autoregressive sampling

### Key Insights:
- GPT = Stack of transformer blocks
- Weight tying saves parameters
- Causal masking prevents cheating
- Scale matters (but small models can still work!)

### Next Steps:
👉 **Lesson 2**: Train this GPT on real data!

You built GPT from scratch! 🎉""")
]

save_notebook("phase4_build_your_gpt/01_gpt_architecture.ipynb", phase4_lesson1)

# ============================================================================
# PHASE 4, LESSON 2: TRAINING YOUR GPT
# ============================================================================
print("\n📝 Creating Phase 4, Lesson 2: Training Your GPT...")

phase4_lesson2 = [
    md("""# Phase 4, Lesson 2: Training Your GPT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suraaj3poudel/Learn-To-Make-GPT-Model/blob/main/phase4_build_your_gpt/02_training_your_gpt.ipynb)

Train your GPT model on real text! 📚

## What You'll Learn
1. Data preparation and tokenization
2. Training loop with AdamW optimizer
3. Gradient accumulation and mixed precision
4. Saving and loading checkpoints
5. Evaluating perplexity

Let's train it!"""),

    code("""# Setup
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Import GPT from previous lesson (you'd normally save it in a separate file)
print('✅ Ready to train GPT!')"""),

    md("""## 1. Prepare Training Data

We'll use a simple text dataset. In practice, you'd use much more data!"""),

    code("""# Sample training data (in practice, use much more!)
text_data = \"\"\"
Artificial intelligence and machine learning are transforming technology.
Neural networks learn patterns from data through training.
Deep learning uses multiple layers to extract features.
Transformers revolutionized natural language processing.
GPT models generate coherent and contextual text.
Attention mechanisms allow models to focus on relevant information.
Language models predict the next word in a sequence.
Training requires large datasets and computational power.
Embeddings represent words as dense vectors.
Self-attention computes relationships between all tokens.
Fine-tuning adapts pre-trained models to specific tasks.
Modern AI systems can understand and generate human language.
Text generation creates meaningful and fluent output.
Machine translation converts text between languages.
Question answering systems extract information from text.
\"\"\" * 20  # Repeat for more data

print(f"Training text length: {len(text_data)} characters")
print(f"Sample: {text_data[:200]}...")"""),

    md("""## 2. Tokenization

Convert text to tokens. We'll use simple character-level tokenization."""),

    code("""class CharTokenizer:
    def __init__(self, text):
        # Get unique characters
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # Create mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])

# Create tokenizer
tokenizer = CharTokenizer(text_data)
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Vocabulary: {''.join(list(tokenizer.char_to_idx.keys())[:50])}...")

# Encode data
encoded_data = torch.tensor(tokenizer.encode(text_data), dtype=torch.long)
print(f"\\nEncoded data shape: {encoded_data.shape}")
print(f"First 50 tokens: {encoded_data[:50]}")"""),

    md("""## 3. Create Dataset

PyTorch Dataset for training"""),

    code("""class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Get chunk of data
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]  # Input
        y = chunk[1:]   # Target (shifted by 1)
        return x, y

# Create dataset and dataloader
block_size = 64
batch_size = 32

dataset = TextDataset(encoded_data, block_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"Dataset size: {len(dataset)}")
print(f"Batch size: {batch_size}")
print(f"Number of batches: {len(dataloader)}")

# Test batch
x, y = next(iter(dataloader))
print(f"\\nBatch input shape: {x.shape}")
print(f"Batch target shape: {y.shape}")"""),

    md("""## 4. Create Model

Small GPT model for our dataset"""),

    code("""# Import GPT class from Lesson 1
# (In practice, you'd import from a separate file)

from types import SimpleNamespace

# Simple config
config = SimpleNamespace(
    vocab_size=tokenizer.vocab_size,
    n_positions=block_size,
    n_embd=128,
    n_layer=4,
    n_head=4,
    dropout=0.1,
)

# We'll use the GPT class from Lesson 1
# For this demo, let's create a simplified version

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4*n_embd,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            ) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)
        
        # Create causal mask
        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()
        
        x = self.blocks(x, src_mask=mask, is_causal=True)
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

# Create model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleGPT(
    vocab_size=tokenizer.vocab_size,
    n_embd=config.n_embd,
    n_head=config.n_head,
    n_layer=config.n_layer,
    block_size=block_size,
    dropout=config.dropout
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model created with {n_params:,} parameters")
print(f"Device: {device}")"""),

    md("""## 5. Training Loop

Train the model!"""),

    code("""# Training configuration
learning_rate = 3e-4
num_epochs = 10

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
losses = []
model.train()

print("Training...\\n")
for epoch in range(num_epochs):
    epoch_loss = 0
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(losses, marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

print("\\n✅ Training complete!")"""),

    md("""## 6. Text Generation

Generate text from the trained model!"""),

    code("""@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0):
    model.eval()
    
    # Encode prompt
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    # Generate
    for _ in range(max_new_tokens):
        # Crop to block size
        idx_cond = idx if idx.size(1) <= model.block_size else idx[:, -model.block_size:]
        
        # Forward pass
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append
        idx = torch.cat([idx, idx_next], dim=1)
    
    # Decode
    return tokenizer.decode(idx[0].tolist())

# Generate text
prompts = [
    "Artificial intelligence",
    "Machine learning",
    "Neural networks",
]

print("\\nGenerated text:\\n")
print("=" * 60)

for prompt in prompts:
    generated = generate(model, tokenizer, prompt, max_new_tokens=150, temperature=0.8)
    print(f"Prompt: '{prompt}'")
    print(f"Generated:\\n{generated}\\n")
    print("-" * 60)

print("\\n✅ Generation complete!")"""),

    md("""## 7. Save and Load Model

Save your trained model!"""),

    code("""# Save checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
    'tokenizer_vocab': tokenizer.char_to_idx,
    'losses': losses,
}

torch.save(checkpoint, 'gpt_checkpoint.pt')
print("✅ Model saved to 'gpt_checkpoint.pt'")

# Load checkpoint
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    # Recreate model
    model = SimpleGPT(
        vocab_size=len(checkpoint['tokenizer_vocab']),
        n_embd=checkpoint['config'].n_embd,
        n_head=checkpoint['config'].n_head,
        n_layer=checkpoint['config'].n_layer,
        block_size=checkpoint['config'].n_positions,
        dropout=checkpoint['config'].dropout
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint

print("\\nTo load:")
print("model, checkpoint = load_model('gpt_checkpoint.pt')")"""),

    md("""## Summary

### What We Did:
1. **Prepared data** - Text to tokens
2. **Created dataset** - PyTorch DataLoader
3. **Trained GPT** - Full training loop
4. **Generated text** - Autoregressive sampling
5. **Saved model** - Checkpointing

### Key Insights:
- More data = better results
- Larger models need more compute
- Temperature controls creativity
- Regular checkpointing is essential

### Next Steps:
👉 **Lesson 3**: Build a chat interface with Gradio!

### Improvements to Try:
- Train on larger text corpus
- Increase model size
- Add learning rate scheduling
- Use gradient accumulation
- Implement early stopping

You trained your own GPT! 🚀""")
]

save_notebook("phase4_build_your_gpt/02_training_your_gpt.ipynb", phase4_lesson2)

# ============================================================================
# PHASE 4, LESSON 3: CHAT INTERFACE
# ============================================================================
print("\n📝 Creating Phase 4, Lesson 3: Chat Interface with Gradio...")

phase4_lesson3 = [
    md("""# Phase 4, Lesson 3: Chat Interface with Gradio

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suraaj3poudel/Learn-To-Make-GPT-Model/blob/main/phase4_build_your_gpt/03_chat_interface.ipynb)

Build an interactive chat interface for your GPT! 💬

## What You'll Learn
1. Using Gradio for UI
2. Chat-style interactions
3. Prompt engineering
4. Temperature and sampling controls
5. Deploying your chatbot

The final wow moment! 🎉"""),

    code("""# Setup
import torch
import torch.nn.functional as F
import gradio as gr

print('✅ Ready to build a chat interface!')
print('Make sure you have gradio installed: pip install gradio')"""),

    md("""## 1. Load Trained Model

First, load your trained GPT model from Lesson 2."""),

    code("""# For this demo, we'll create a mock model
# In practice, load your actual trained model

class MockGPT:
    \"\"\"Mock GPT for demonstration\"\"\"
    def __init__(self):
        self.name = "Mini-GPT"
    
    def generate(self, prompt, max_length=100, temperature=0.8):
        # This is a mock - replace with your actual model
        responses = [
            f"Based on '{prompt}', I would say that machine learning is fascinating!",
            f"That's an interesting question about '{prompt}'. Let me think...",
            f"Regarding '{prompt}', here's what I know: AI is transforming technology.",
        ]
        import random
        return random.choice(responses) + " " + prompt

# In practice, replace with:
# model = load_model('gpt_checkpoint.pt')

model = MockGPT()
print("✅ Model loaded (using mock for demo)")"""),

    md("""## 2. Create Generation Function

Function to generate responses from the model."""),

    code("""def generate_response(prompt, max_length=150, temperature=0.8, top_k=50):
    \"\"\"
    Generate response from the model
    
    Args:
        prompt: User input text
        max_length: Maximum tokens to generate
        temperature: Sampling temperature (higher = more creative)
        top_k: Only sample from top k tokens
    
    Returns:
        Generated text
    \"\"\"
    if not prompt.strip():
        return "Please enter a message!"
    
    # Generate
    response = model.generate(
        prompt=prompt,
        max_length=max_length,
        temperature=temperature
    )
    
    return response

# Test
test_prompt = "What is machine learning?"
response = generate_response(test_prompt)
print(f"Prompt: {test_prompt}")
print(f"Response: {response}")"""),

    md("""## 3. Build Gradio Interface

Create a beautiful chat interface!"""),

    code("""def chat_interface(message, history, temperature, max_length):
    \"\"\"
    Chat function for Gradio
    
    Args:
        message: User's message
        history: Chat history
        temperature: Sampling temperature
        max_length: Max tokens to generate
    
    Returns:
        Updated history
    \"\"\"
    # Generate response
    response = generate_response(
        prompt=message,
        max_length=max_length,
        temperature=temperature
    )
    
    # Add to history
    history.append((message, response))
    
    return history

# Create Gradio interface
with gr.Blocks(title="GPT Chat Interface") as demo:
    gr.Markdown(
        \"\"\"
        # 🤖 Chat with Your GPT Model!
        
        This is your custom-built GPT chatbot. Try asking questions!
        \"\"\"
    )
    
    chatbot = gr.Chatbot(
        label="Chat History",
        height=400
    )
    
    with gr.Row():
        msg = gr.Textbox(
            label="Your Message",
            placeholder="Type your message here...",
            scale=4
        )
        send = gr.Button("Send", scale=1)
    
    with gr.Accordion("⚙️ Advanced Settings", open=False):
        temperature = gr.Slider(
            minimum=0.1,
            maximum=2.0,
            value=0.8,
            step=0.1,
            label="Temperature (Higher = More Creative)"
        )
        max_length = gr.Slider(
            minimum=50,
            maximum=500,
            value=150,
            step=10,
            label="Max Length"
        )
    
    clear = gr.Button("Clear Chat")
    
    # Event handlers
    msg.submit(chat_interface, [msg, chatbot, temperature, max_length], chatbot)
    msg.submit(lambda: "", None, msg)  # Clear input
    
    send.click(chat_interface, [msg, chatbot, temperature, max_length], chatbot)
    send.click(lambda: "", None, msg)  # Clear input
    
    clear.click(lambda: None, None, chatbot)
    
    gr.Markdown(
        \"\"\"
        ### 💡 Tips:
        - Adjust temperature for more/less creative responses
        - Higher max length allows longer responses
        - Try different prompts to see what your model learned!
        \"\"\"
    )

print("\\n✅ Gradio interface created!")
print("\\nTo launch, run:")
print("  demo.launch()")"""),

    md("""## 4. Launch the Interface

Run this cell to start the chat interface!"""),

    code("""# Launch the interface
# Uncomment to run:

# demo.launch(
#     share=False,      # Set True to create public link
#     server_name="0.0.0.0",  # Allow external connections
#     server_port=7860
# )

print("Run demo.launch() to start the chat interface!")
print("\\nOptions:")
print("  - share=True: Create a public link (for demos)")
print("  - share=False: Local only (more secure)")"""),

    md("""## 5. Prompt Engineering

Get better responses with good prompts!"""),

    code("""# Example prompts
good_prompts = [
    # Specific questions
    "Explain how transformers work in 2-3 sentences.",
    
    # Context setting
    "As an AI expert, what are the key components of a neural network?",
    
    # Format instructions
    "List 3 benefits of machine learning:\\n1.",
    
    # Role playing
    "You are a helpful AI assistant. Explain deep learning simply.",
]

poor_prompts = [
    # Too vague
    "Tell me stuff",
    
    # No context
    "It",
    
    # Too broad
    "Everything about AI",
]

print("Good Prompts:")
for i, prompt in enumerate(good_prompts, 1):
    print(f"{i}. {prompt}")

print("\\n\\nPoor Prompts (avoid these):")
for i, prompt in enumerate(poor_prompts, 1):
    print(f"{i}. {prompt}")

print("\\n💡 Better prompts = better responses!")"""),

    md("""## 6. Advanced Features

Add more functionality to your chatbot!"""),

    code("""def enhanced_chat(message, history, system_prompt="You are a helpful AI assistant."):
    \"\"\"
    Enhanced chat with system prompt
    
    Args:
        message: User message
        history: Chat history
        system_prompt: System-level instruction
    \"\"\"
    # Combine system prompt with user message
    full_prompt = f"{system_prompt}\\n\\nUser: {message}\\nAssistant:"
    
    # Generate
    response = generate_response(full_prompt, max_length=200, temperature=0.7)
    
    return response

# Example system prompts
system_prompts = {
    "Helpful Assistant": "You are a helpful, friendly AI assistant.",
    "Code Helper": "You are an expert programmer who explains code clearly.",
    "Creative Writer": "You are a creative writer who crafts engaging stories.",
    "Teacher": "You are a patient teacher who explains concepts simply.",
}

print("System Prompt Examples:\\n")
for name, prompt in system_prompts.items():
    print(f"{name}: {prompt}")

print("\\n💡 System prompts shape the model's behavior!")"""),

    md("""## 7. Deployment Options

Ways to share your chatbot with others!"""),

    code("""# Deployment options
deployment_options = {
    "Local": {
        "command": "demo.launch(share=False)",
        "pros": "Fast, secure, free",
        "cons": "Only you can access",
    },
    "Gradio Share": {
        "command": "demo.launch(share=True)",
        "pros": "Creates public link, easy to share",
        "cons": "Temporary link (72 hours)",
    },
    "Hugging Face Spaces": {
        "command": "Upload to HF Spaces",
        "pros": "Permanent, professional, free tier",
        "cons": "Requires account setup",
    },
    "Cloud Server": {
        "command": "Deploy to AWS/GCP/Azure",
        "pros": "Full control, scalable",
        "cons": "Costs money, more setup",
    },
}

print("Deployment Options:\\n")
for option, details in deployment_options.items():
    print(f"📌 {option}")
    print(f"   Command: {details['command']}")
    print(f"   Pros: {details['pros']}")
    print(f"   Cons: {details['cons']}")
    print()"""),

    md("""## 8. Complete Example

Full working chatbot with all features!"""),

    code("""# Complete chatbot implementation
def create_full_chatbot(model):
    \"\"\"Create complete chatbot with all features\"\"\"
    
    with gr.Blocks(theme=gr.themes.Soft(), title="My GPT Chatbot") as app:
        gr.Markdown("# 🤖 Your Custom GPT Chatbot")
        gr.Markdown("Built from scratch in this ML learning journey!")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, label="Chat")
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask me anything...",
                        show_label=False,
                        scale=4
                    )
                    submit = gr.Button("Send 🚀", scale=1)
                
                clear = gr.ClearButton([msg, chatbot])
            
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Settings")
                
                temperature = gr.Slider(
                    0.1, 2.0, value=0.8, step=0.1,
                    label="Temperature"
                )
                
                max_length = gr.Slider(
                    50, 500, value=150, step=50,
                    label="Max Length"
                )
                
                system_prompt = gr.Dropdown(
                    choices=list(system_prompts.keys()),
                    value="Helpful Assistant",
                    label="Role"
                )
        
        gr.Markdown(
            \"\"\"
            ---
            ### 📊 Model Info
            - Architecture: Transformer (GPT-style)
            - Parameters: Custom trained
            - Training: Phase 1-4 curriculum
            
            ### 🎯 Tips
            - Be specific in your questions
            - Adjust temperature for creativity
            - Try different roles for varied responses
            \"\"\"
        )
        
        # Connect events
        msg.submit(chat_interface, [msg, chatbot, temperature, max_length], chatbot)
        submit.click(chat_interface, [msg, chatbot, temperature, max_length], chatbot)
    
    return app

print("\\n✅ Complete chatbot ready!")
print("Run: app = create_full_chatbot(model)")
print("Then: app.launch()")"""),

    md("""## Summary

### What We Built:
1. **Chat Interface** - Beautiful Gradio UI
2. **Generation Controls** - Temperature, length, etc.
3. **Prompt Engineering** - Better inputs = better outputs
4. **System Prompts** - Shape model behavior
5. **Deployment Options** - Share with the world!

### Key Insights:
- UI makes models accessible
- Prompt engineering is crucial
- Temperature controls creativity
- System prompts set behavior
- Many ways to deploy

### 🎉 CONGRATULATIONS! 🎉

You've completed the entire journey:
- ✅ Phase 1: Neural Networks
- ✅ Phase 2: Text & Embeddings
- ✅ Phase 3: Transformers
- ✅ Phase 4: GPT & Chat Interface

You built GPT from scratch and created a chatbot! 

### What's Next?
- Train on larger datasets
- Experiment with bigger models
- Try fine-tuning on specific tasks
- Deploy and share your chatbot
- Keep learning and building!

### Resources:
- Hugging Face Transformers: https://huggingface.co/transformers/
- OpenAI GPT Papers: https://openai.com/research/
- Attention Is All You Need: https://arxiv.org/abs/1706.03762

**You're now a GPT builder! Keep exploring! 🚀**""")
]

save_notebook("phase4_build_your_gpt/03_chat_interface.ipynb", phase4_lesson3)

print("\n" + "="*60)
print("🎉 ALL PHASE 4 NOTEBOOKS CREATED! 🎉")
print("="*60)
print("\n✅ Complete ML Learning Journey:")
print("   Phase 1: Neural Networks ✅")
print("   Phase 2: Text & Embeddings ✅")
print("   Phase 3: Mini Transformer ✅")
print("   Phase 4: Build Your GPT ✅")
print("\n🚀 You now have a complete GPT learning curriculum!")
print("\nNext steps:")
print("  1. Open Jupyter: jupyter notebook")
print("  2. Work through each phase in order")
print("  3. Build amazing AI projects!")
print("\n" + "="*60)
