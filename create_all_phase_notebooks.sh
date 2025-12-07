#!/bin/bash
# Script to create all Phase 2-4 notebooks with comprehensive content

echo "🚀 Creating all Phase 2-4 notebooks..."
echo "This will generate 8 complete learning notebooks!"
echo ""

cd "/Users/poudels2/Desktop/GPT Project"

# Run Python script to generate all notebooks
python3 << 'PYTHON_SCRIPT'
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

# ============================================================================
# PHASE 2, LESSON 2: SENTIMENT ANALYSIS
# ============================================================================
print("\n📝 Creating Phase 2, Lesson 2: Sentiment Analysis...")

phase2_lesson2 = [
    md("""# Phase 2, Lesson 2: Sentiment Analysis with Embeddings

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suraaj3poudel/Learn-To-Make-GPT-Model/blob/main/phase2_text_and_embeddings/02_sentiment_analysis.ipynb)

Build your first NLP classifier! 🎭

## What You'll Learn
1. Sentiment analysis task
2. Using embeddings for classification
3. Building a neural network text classifier
4. Evaluating your model

Let's build something useful!"""),

    code("""# Setup
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

print('✅ Ready to build a sentiment analyzer!')"""),

    md("""## 1. The Task: Sentiment Analysis

**Goal**: Classify text as POSITIVE or NEGATIVE

Examples:
- "I love this movie!" → POSITIVE ✅
- "This is terrible" → NEGATIVE ❌
- "Best day ever!" → POSITIVE ✅

This is a **classification** problem!"""),

    code("""# Simple sentiment dataset
reviews = [
    ("I love this product", "positive"),
    ("This is amazing", "positive"),
    ("Best purchase ever", "positive"),
    ("Absolutely wonderful", "positive"),
    ("Great quality", "positive"),
    ("Terrible experience", "negative"),
    ("Waste of money", "negative"),
    ("Very disappointed", "negative"),
    ("Poor quality", "negative"),
    ("Do not buy this", "negative"),
    ("Pretty good", "positive"),
    ("Not bad", "positive"),
    ("Could be better", "negative"),
    ("Not worth it", "negative"),
]

print(f"Dataset size: {len(reviews)} reviews")
print("\\nSample reviews:")
for review, sentiment in reviews[:5]:
    print(f"  '{review}' → {sentiment}")"""),

    md("""## 2. Prepare Data: Tokenization & Vocabulary

Same as before - create vocabulary and encode text."""),

    code("""# Build vocabulary from all reviews
all_words = []
for review, _ in reviews:
    words = re.findall(r'\\w+', review.lower())
    all_words.extend(words)

# Create vocabulary
vocab = {word: i for i, word in enumerate(sorted(set(all_words)))}
vocab_size = len(vocab)

# Add special tokens
vocab['<UNK>'] = len(vocab)  # Unknown words

print(f"Vocabulary size: {vocab_size + 1}")
print(f"\\nVocabulary: {list(vocab.keys())[:15]}...")

# Encode reviews
def encode_review(review, vocab):
    words = re.findall(r'\\w+', review.lower())
    return [vocab.get(word, vocab['<UNK>']) for word in words]

# Test encoding
test_review = "I love this product"
encoded = encode_review(test_review, vocab)
print(f"\\nEncoded '{test_review}': {encoded}")"""),

    md("""## 3. Create Embeddings

Each word gets a learned vector representation."""),

    code("""# Initialize embeddings
embedding_dim = 10  # Small for our tiny dataset
np.random.seed(42)

# Embedding matrix: each word → vector
embeddings = np.random.randn(len(vocab), embedding_dim) * 0.1

print(f"Embedding matrix shape: {embeddings.shape}")
print(f"Each word is represented by a {embedding_dim}-dimensional vector")

# Example: Get embedding for a word
word = 'love'
if word in vocab:
    word_idx = vocab[word]
    word_vector = embeddings[word_idx]
    print(f"\\n'{word}' embedding: {word_vector}")"""),

    md("""## 4. Build the Classifier

Simple architecture:
1. **Embedding layer**: Words → Vectors
2. **Averaging**: Average all word vectors
3. **Dense layer**: Make prediction"""),

    code("""class SentimentClassifier:
    def __init__(self, vocab_size, embedding_dim):
        # Embedding layer
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
        
        # Classification layer
        self.W = np.random.randn(embedding_dim, 1) * 0.1
        self.b = np.zeros(1)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, word_indices):
        # Get embeddings for all words
        word_vecs = self.embeddings[word_indices]
        
        # Average word vectors (simple but effective!)
        avg_vec = np.mean(word_vecs, axis=0)
        
        # Classification
        logit = np.dot(avg_vec, self.W) + self.b
        prob = self.sigmoid(logit)
        
        return prob[0], avg_vec
    
    def predict(self, word_indices):
        prob, _ = self.forward(word_indices)
        return "positive" if prob > 0.5 else "negative"

# Create model
model = SentimentClassifier(len(vocab), embedding_dim)

# Test prediction (before training)
test_indices = encode_review("I love this", vocab)
prediction = model.predict(test_indices)
print(f"Prediction (before training): {prediction}")
print("(Random guess - we haven't trained yet!)")"""),

    md("""## 5. Training the Model

Use gradient descent to learn good embeddings and weights!"""),

    code("""def train_model(model, reviews, epochs=200, lr=0.1):
    losses = []
    
    # Encode labels: positive=1, negative=0
    encoded_reviews = []
    labels = []
    for review, sentiment in reviews:
        indices = encode_review(review, vocab)
        encoded_reviews.append(indices)
        labels.append(1.0 if sentiment == "positive" else 0.0)
    
    print("Training...")
    for epoch in range(epochs):
        epoch_loss = 0
        
        for indices, label in zip(encoded_reviews, labels):
            # Forward pass
            prob, avg_vec = model.forward(indices)
            
            # Compute loss (binary cross-entropy)
            loss = -label * np.log(prob + 1e-10) - (1-label) * np.log(1-prob + 1e-10)
            epoch_loss += loss
            
            # Backward pass (simplified)
            error = prob - label
            
            # Update weights
            model.W -= lr * np.outer(avg_vec, error)
            model.b -= lr * error
            
            # Update embeddings (simplified)
            grad_embed = np.outer(model.W, error).T / len(indices)
            for idx in indices:
                model.embeddings[idx] -= lr * grad_embed
        
        losses.append(epoch_loss / len(reviews))
        
        if (epoch + 1) % 40 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {losses[-1]:.4f}")
    
    return losses

# Train!
losses = train_model(model, reviews, epochs=200, lr=0.1)

# Plot loss
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

print("\\n✅ Training complete!")"""),

    md("""## 6. Test the Trained Model

Let's see how well it works!"""),

    code("""# Test on training data
correct = 0
for review, true_label in reviews:
    indices = encode_review(review, vocab)
    pred_label = model.predict(indices)
    is_correct = pred_label == true_label
    correct += is_correct
    print(f"'{review}' → Predicted: {pred_label}, True: {true_label} {'✅' if is_correct else '❌'}")

accuracy = correct / len(reviews) * 100
print(f"\\nAccuracy: {accuracy:.1f}%")"""),

    md("""## 7. Try Your Own Reviews!

Test the model on new text!"""),

    code("""# Test on new reviews (not in training data)
new_reviews = [
    "This is fantastic",
    "Horrible product",
    "Really great",
    "Very bad",
    "Absolutely love it",
]

print("Testing on new reviews:\\n")
for review in new_reviews:
    indices = encode_review(review, vocab)
    prediction = model.predict(indices)
    prob, _ = model.forward(indices)
    print(f"'{review}'")
    print(f"  → {prediction.upper()} (confidence: {prob:.2%})\\n")"""),

    md("""## Summary

### What We Built:
1. **Sentiment classifier** using word embeddings
2. **Embedding layer** that learns meaningful word vectors
3. **Simple averaging** to combine word vectors
4. **Binary classifier** for positive/negative

### Key Insights:
- Embeddings make words meaningful to neural networks
- Averaging word vectors is simple but effective
- The model learns which words indicate positive/negative

### Next Steps:
👉 **Lesson 3**: Learn about **attention** - a more sophisticated way to combine word vectors!

Great work! You built your first NLP classifier! 🎉""")
]

save_notebook("phase2_text_and_embeddings/02_sentiment_analysis.ipynb", phase2_lesson2)

# ============================================================================
# PHASE 2, LESSON 3: ATTENTION MECHANISM
# ============================================================================
print("\\n📝 Creating Phase 2, Lesson 3: Attention Mechanism...")

phase2_lesson3 = [
    md("""# Phase 2, Lesson 3: Attention Mechanism

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suraaj3poudel/Learn-To-Make-GPT-Model/blob/main/phase2_text_and_embeddings/03_attention_mechanism.ipynb)

The breakthrough that powers GPT! 🧠

## What You'll Learn
1. Why attention is needed
2. How attention works
3. Self-attention mechanism
4. Building attention from scratch

This is THE key to modern AI!"""),

    code("""# Setup
import numpy as np
import matplotlib.pyplot as plt

print('✅ Ready to learn attention!')"""),

    md("""## 1. The Problem with Averaging

In Lesson 2, we **averaged** all word vectors:

```python
sentence = "I love this amazing product"
avg = mean([vec_I, vec_love, vec_this, vec_amazing, vec_product])
```

**Problem**: All words get equal weight!

But some words are more important:
- "I **love** this **amazing** product" 
- The words "love" and "amazing" matter most for sentiment!

We need a way to **focus** on important words. That's **attention**!"""),

    md("""## 2. Attention: The Core Idea

**Attention** = Learn which words to focus on

Instead of:
```
output = mean(all_words)
```

We want:
```
output = 0.05 * word1 + 0.10 * word2 + 0.60 * word3 + ...
```

Where the weights (0.05, 0.10, 0.60, ...) are **learned**!"""),

    code("""# Simple example: Manual attention weights
sentence = ["I", "love", "this", "amazing", "product"]
word_vecs = np.random.randn(5, 10)  # 5 words, 10-dim vectors

# Manual attention weights (what we want to learn)
attention_weights = np.array([0.05, 0.35, 0.10, 0.40, 0.10])

# Weighted sum (instead of average)
attended_output = np.sum(attention_weights[:, None] * word_vecs, axis=0)

print("Sentence:", " ".join(sentence))
print("\\nAttention weights:")
for word, weight in zip(sentence, attention_weights):
    print(f"  {word}: {weight:.2f} {'🔥' if weight > 0.3 else ''}")

print(f"\\nAttended output shape: {attended_output.shape}")
print("The model is 'paying attention' to 'love' and 'amazing'!")"""),

    md("""## 3. How to Compute Attention Weights?

**Key insight**: Use the vectors themselves to decide importance!

**Formula** (simplified):
1. **Score**: How relevant is each word to what we're looking for?
2. **Softmax**: Convert scores to weights that sum to 1
3. **Weighted sum**: Combine vectors using weights

Let's build it!"""),

    code("""def simple_attention(word_vectors, query_vector):
    \"\"\"
    Compute attention weights and output
    
    Args:
        word_vectors: (num_words, embedding_dim) - the words to attend to
        query_vector: (embedding_dim,) - what we're looking for
    \"\"\"
    # Step 1: Compute scores (dot product)
    scores = np.dot(word_vectors, query_vector)
    
    # Step 2: Softmax to get weights
    exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
    weights = exp_scores / exp_scores.sum()
    
    # Step 3: Weighted sum
    output = np.sum(weights[:, None] * word_vectors, axis=0)
    
    return output, weights

# Example
words = np.random.randn(5, 10)  # 5 words, 10 dimensions
query = np.random.randn(10)      # What we're looking for

output, weights = simple_attention(words, query)

print("Attention weights:", weights)
print(f"Sum of weights: {weights.sum():.4f} (should be 1.0)")
print(f"\\nOutput shape: {output.shape}")"""),

    md("""## 4. Self-Attention: The Magic Ingredient

**Self-attention** = Each word attends to all other words (including itself)!

For the sentence "I love programming":
- "I" attends to ["I", "love", "programming"]
- "love" attends to ["I", "love", "programming"]  
- "programming" attends to ["I", "love", "programming"]

Each word gets its own unique representation based on context!"""),

    code("""def self_attention(word_vectors):
    \"\"\"
    Self-attention: each word attends to all words
    
    Args:
        word_vectors: (num_words, embedding_dim)
    
    Returns:
        outputs: (num_words, embedding_dim) - attended representations
        all_weights: (num_words, num_words) - attention matrix
    \"\"\"
    num_words = word_vectors.shape[0]
    outputs = []
    all_weights = []
    
    # For each word as query
    for i in range(num_words):
        query = word_vectors[i]
        
        # Attend to all words (including itself)
        output, weights = simple_attention(word_vectors, query)
        outputs.append(output)
        all_weights.append(weights)
    
    return np.array(outputs), np.array(all_weights)

# Example
sentence_vecs = np.random.randn(4, 8)  # 4 words, 8 dimensions
attended_vecs, attention_matrix = self_attention(sentence_vecs)

print(f"Input shape: {sentence_vecs.shape}")
print(f"Output shape: {attended_vecs.shape}")
print(f"Attention matrix shape: {attention_matrix.shape}")
print(f"\\nAttention matrix:\\n{attention_matrix}")"""),

    md("""## 5. Visualizing Attention

Let's see which words attend to which!"""),

    code("""# Create a simple sentence
sentence = ["I", "love", "machine", "learning"]
num_words = len(sentence)

# Create word vectors (simplified)
np.random.seed(42)
vecs = np.random.randn(num_words, 10)

# Compute self-attention
_, attention_matrix = self_attention(vecs)

# Visualize
plt.figure(figsize=(8, 6))
plt.imshow(attention_matrix, cmap='Blues')
plt.colorbar(label='Attention Weight')
plt.xticks(range(num_words), sentence)
plt.yticks(range(num_words), sentence)
plt.xlabel('Attending to...')
plt.ylabel('Word')
plt.title('Self-Attention Matrix')

# Add values
for i in range(num_words):
    for j in range(num_words):
        text = plt.text(j, i, f'{attention_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=10)

plt.tight_layout()
plt.show()

print("Each row shows where that word 'pays attention'")
print("Darker = more attention")"""),

    md("""## 6. Scaled Dot-Product Attention (Real Version)

The real attention mechanism used in Transformers!

**Formula**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- **Q** (Query): What am I looking for?
- **K** (Key): What do I contain?
- **V** (Value): What do I actually output?
- **d_k**: Dimension (for scaling)

Let's implement it!"""),

    code("""def scaled_dot_product_attention(Q, K, V):
    \"\"\"
    Scaled dot-product attention (used in Transformers!)
    
    Args:
        Q: (num_queries, d_k) - Queries
        K: (num_keys, d_k) - Keys
        V: (num_values, d_v) - Values
    
    Returns:
        output: (num_queries, d_v)
        attention_weights: (num_queries, num_keys)
    \"\"\"
    d_k = Q.shape[-1]
    
    # Compute attention scores
    scores = np.dot(Q, K.T) / np.sqrt(d_k)  # Scaling is important!
    
    # Softmax to get weights
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    
    # Apply attention to values
    output = np.dot(attention_weights, V)
    
    return output, attention_weights

# Example: 4 words, 8-dimensional embeddings
num_words = 4
d_model = 8

np.random.seed(42)
embeddings = np.random.randn(num_words, d_model)

# In self-attention: Q = K = V = embeddings
output, weights = scaled_dot_product_attention(embeddings, embeddings, embeddings)

print(f"Input shape: {embeddings.shape}")
print(f"Output shape: {output.shape}")
print(f"\\nAttention weights shape: {weights.shape}")
print(f"\\nAttention weights:\\n{weights}")
print(f"\\nEach row sums to: {weights.sum(axis=1)}")"""),

    md("""## 7. Why Scaling Matters

The division by √d_k is crucial!

**Without scaling**: Dot products get very large → softmax saturates → gradients vanish

Let's see the difference:"""),

    code("""# Compare with and without scaling
d_k = 64  # Typical dimension

Q = np.random.randn(3, d_k)
K = np.random.randn(3, d_k)

# Without scaling
scores_no_scale = np.dot(Q, K.T)
print("Without scaling:")
print(f"  Scores: {scores_no_scale[0]}")
print(f"  Range: [{scores_no_scale.min():.2f}, {scores_no_scale.max():.2f}]")

# With scaling
scores_scaled = np.dot(Q, K.T) / np.sqrt(d_k)
print(f"\\nWith scaling:")
print(f"  Scores: {scores_scaled[0]}")
print(f"  Range: [{scores_scaled.min():.2f}, {scores_scaled.max():.2f}]")

print(f"\\n✅ Scaling keeps values in a reasonable range!")"""),

    md("""## 8. Multi-Head Attention (Preview)

**Multi-head attention** = Run multiple attention mechanisms in parallel!

Why?
- Different heads can focus on different aspects
- Head 1 might focus on syntax
- Head 2 might focus on semantics
- Head 3 might focus on long-range dependencies

We'll implement this fully in Phase 3!"""),

    code("""# Conceptual example
num_heads = 3
d_model = 12
d_k = d_model // num_heads  # Each head gets smaller dimension

print(f"Model dimension: {d_model}")
print(f"Number of heads: {num_heads}")
print(f"Dimension per head: {d_k}")

print("\\nEach head learns to attend to different patterns!")
print("Then we concatenate all heads and project back to d_model.")"""),

    md("""## Summary

### What We Learned:
1. **Attention** = Learn what to focus on
2. **Self-attention** = Each word attends to all words
3. **Scaled dot-product** = The real attention formula
4. **Q, K, V** = Query, Key, Value matrices
5. **Scaling** = Keeps gradients stable

### Key Insights:
- Attention replaces fixed averaging
- It's **learned** - the model decides what's important
- Powers all modern language models (GPT, BERT, etc.)

### Next Steps:
👉 **Phase 3**: Build a complete **Transformer** using attention!

You now understand the core of modern AI! 🚀""")
]

save_notebook("phase2_text_and_embeddings/03_attention_mechanism.ipynb", phase2_lesson3)

print("\\n" + "="*60)
print("✅ Phase 2 complete! Now creating Phase 3...")
print("="*60)

PYTHON_SCRIPT

echo ""
echo "✅ Phase 2 notebooks created!"
echo "Continuing with Phases 3 and 4..."
