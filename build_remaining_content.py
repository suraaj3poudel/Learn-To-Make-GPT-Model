"""
Script to generate comprehensive notebook content for Phases 2-4
This will create all the learning materials you need!
"""

import json
import os

def create_notebook_cells(phase, lesson_num, title, cells_content):
    """Helper to create notebook structure"""
    filepath = f"/Users/poudels2/Desktop/GPT Project/{phase}/{lesson_num}_{title}.ipynb"
    
    notebook = {
        "cells": cells_content,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.9.6"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    with open(filepath, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Created: {filepath}")

# Phase 2 Notebook 1: Text Processing and Embeddings
phase2_lesson1_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Phase 2, Lesson 1: Text Processing and Embeddings\\n",
            "\\n",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suraaj3poudel/Learn-To-Make-GPT-Model/blob/main/phase2_text_and_embeddings/01_text_processing_and_embeddings.ipynb)\\n",
            "\\n",
            "Welcome to Phase 2! 🎉\\n",
            "\\n",
            "## What You'll Learn\\n",
            "1. Tokenization - Breaking text into pieces\\n",
            "2. Word Embeddings - Converting words to meaningful vectors\\n",
            "3. Word2Vec - Learning word relationships\\n",
            "4. Semantic similarity\\n",
            "\\n",
            "Let's go!"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Setup\\n",
            "import numpy as np\\n",
            "import matplotlib.pyplot as plt\\n",
            "from collections import Counter, defaultdict\\n",
            "import re\\n",
            "\\n",
            "print('✅ Ready to learn about text processing!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Tokenization: Breaking Text into Pieces\\n",
            "\\n",
            "**Tokenization** = splitting text into smaller units (tokens)\\n",
            "\\n",
            "Types:\\n",
            "- **Word tokenization**: Split by spaces\\n",
            "- **Subword tokenization**: Split into meaningful pieces\\n",
            "- **Character tokenization**: Split into individual characters\\n",
            "\\n",
            "Let's try each!"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "text = \\"I love machine learning! It's amazing.\\"\\n",
            "\\n",
            "# Word tokenization (simple)\\n",
            "words = text.lower().split()\\n",
            "print(\\"Words:\\", words)\\n",
            "\\n",
            "# Better: Handle punctuation\\n",
            "words_clean = re.findall(r'\\\\w+', text.lower())\\n",
            "print(\\"\\\\nCleaned words:\\", words_clean)\\n",
            "\\n",
            "# Character tokenization\\n",
            "chars = list(text)\\n",
            "print(\\"\\\\nCharacters:\\", chars[:20], '...')  # First 20"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Vocabulary: Mapping Words to Numbers\\n",
            "\\n",
            "Create a **vocabulary** - a dictionary mapping each unique word to an ID."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Sample corpus\\n",
            "corpus = [\\n",
            "    \\"I love machine learning\\",\\n",
            "    \\"Machine learning is amazing\\",\\n",
            "    \\"I love deep learning\\",\\n",
            "    \\"Deep learning uses neural networks\\",\\n",
            "    \\"Neural networks are powerful\\"\\n",
            "]\\n",
            "\\n",
            "# Tokenize all sentences\\n",
            "all_words = []\\n",
            "for sentence in corpus:\\n",
            "    words = re.findall(r'\\\\w+', sentence.lower())\\n",
            "    all_words.extend(words)\\n",
            "\\n",
            "# Create vocabulary\\n",
            "vocab = {word: i for i, word in enumerate(set(all_words))}\\n",
            "reverse_vocab = {i: word for word, i in vocab.items()}\\n",
            "\\n",
            "print(f\\"Vocabulary size: {len(vocab)}\\")\\n",
            "print(f\\"\\\\nVocabulary: {vocab}\\")\\n",
            "\\n",
            "# Encode a sentence\\n",
            "sentence = \\"I love learning\\"\\n",
            "encoded = [vocab.get(word, -1) for word in re.findall(r'\\\\w+', sentence.lower())]\\n",
            "print(f\\"\\\\nEncoded '{sentence}': {encoded}\\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. One-Hot Encoding (Simple but Limited)\\n",
            "\\n",
            "**One-hot encoding** represents each word as a vector with:\\n",
            "- 1 at the word's position\\n",
            "- 0 everywhere else\\n",
            "\\n",
            "**Problem**: No semantic meaning! 'king' and 'queen' are as different as 'king' and 'apple'."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def one_hot_encode(word, vocab):\\n",
            "    vector = np.zeros(len(vocab))\\n",
            "    if word in vocab:\\n",
            "        vector[vocab[word]] = 1\\n",
            "    return vector\\n",
            "\\n",
            "# Example\\n",
            "word1 = \\"love\\"\\n",
            "word2 = \\"learning\\"\\n",
            "\\n",
            "vec1 = one_hot_encode(word1, vocab)\\n",
            "vec2 = one_hot_encode(word2, vocab)\\n",
            "\\n",
            "print(f\\"{word1}: {vec1}\\")\\n",
            "print(f\\"{word2}: {vec2}\\")\\n",
            "\\n",
            "# Similarity (dot product)\\n",
            "similarity = np.dot(vec1, vec2)\\n",
            "print(f\\"\\\\nSimilarity between '{word1}' and '{word2}': {similarity}\\")\\n",
            "print(\\"Problem: All different words have similarity = 0!\\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Word Embeddings: Dense Meaningful Vectors\\n",
            "\\n",
            "**Word embeddings** = dense vectors (e.g., 100-300 dimensions) that capture meaning!\\n",
            "\\n",
            "Key idea:\\n",
            "- Similar words → similar vectors\\n",
            "- \\"king\\" is closer to \\"queen\\" than to \\"apple\\"\\n",
            "\\n",
            "Let's create simple embeddings!"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Simple embeddings (random initialization - normally we'd train these!)\\n",
            "embedding_dim = 5  # Small for visualization\\n",
            "np.random.seed(42)\\n",
            "\\n",
            "# Create embedding matrix: each word gets a random vector\\n",
            "word_embeddings = {}\\n",
            "for word in vocab:\\n",
            "    word_embeddings[word] = np.random.randn(embedding_dim)\\n",
            "\\n",
            "# Show some embeddings\\n",
            "for word in ['love', 'learning', 'machine']:\\n",
            "    print(f\\"{word}: {word_embeddings[word]}\\")\\n",
            "\\n",
            "# Calculate similarity\\n",
            "def cosine_similarity(v1, v2):\\n",
            "    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))\\n",
            "\\n",
            "sim = cosine_similarity(word_embeddings['machine'], word_embeddings['learning'])\\n",
            "print(f\\"\\\\nSimilarity (machine, learning): {sim:.3f}\\")\\n",
            "print(\\"Note: These are random embeddings. Real ones would show actual semantic similarity!\\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Word2Vec Concept: Learning from Context\\n",
            "\\n",
            "**Word2Vec** learns embeddings based on the idea:\\n",
            "> *\\"You shall know a word by the company it keeps\\"*\\n",
            "\\n",
            "Two approaches:\\n",
            "1. **CBOW** (Continuous Bag of Words): Predict word from context\\n",
            "2. **Skip-gram**: Predict context from word\\n",
            "\\n",
            "Let's implement a simple version!"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create training pairs (word, context)\\n",
            "def create_context_pairs(corpus, window_size=2):\\n",
            "    pairs = []\\n",
            "    \\n",
            "    for sentence in corpus:\\n",
            "        words = re.findall(r'\\\\w+', sentence.lower())\\n",
            "        \\n",
            "        for i, word in enumerate(words):\\n",
            "            # Get context words (within window)\\n",
            "            start = max(0, i - window_size)\\n",
            "            end = min(len(words), i + window_size + 1)\\n",
            "            \\n",
            "            context = []\\n",
            "            for j in range(start, end):\\n",
            "                if j != i:  # Skip the center word\\n",
            "                    context.append(words[j])\\n",
            "            \\n",
            "            for context_word in context:\\n",
            "                pairs.append((word, context_word))\\n",
            "    \\n",
            "    return pairs\\n",
            "\\n",
            "# Generate pairs\\n",
            "training_pairs = create_context_pairs(corpus, window_size=2)\\n",
            "\\n",
            "print(\\"Sample training pairs (word, context):\\")\\n",
            "for pair in training_pairs[:10]:\\n",
            "    print(f\\"  {pair}\\")\\n",
            "\\n",
            "print(f\\"\\\\nTotal training pairs: {len(training_pairs)}\\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Simple Skip-Gram Implementation\\n",
            "\\n",
            "Let's build a tiny Word2Vec model from scratch!"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "class SimpleWord2Vec:\\n",
            "    def __init__(self, vocab_size, embedding_dim):\\n",
            "        # Initialize embeddings randomly\\n",
            "        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01\\n",
            "        self.context_weights = np.random.randn(embedding_dim, vocab_size) * 0.01\\n",
            "        \\n",
            "    def softmax(self, x):\\n",
            "        exp_x = np.exp(x - np.max(x))\\n",
            "        return exp_x / exp_x.sum()\\n",
            "    \\n",
            "    def forward(self, word_idx):\\n",
            "        # Get word embedding\\n",
            "        h = self.embeddings[word_idx]\\n",
            "        # Compute scores\\n",
            "        scores = np.dot(h, self.context_weights)\\n",
            "        # Apply softmax\\n",
            "        probs = self.softmax(scores)\\n",
            "        return h, probs\\n",
            "    \\n",
            "    def train_step(self, word_idx, context_idx, learning_rate=0.01):\\n",
            "        # Forward pass\\n",
            "        h, probs = self.forward(word_idx)\\n",
            "        \\n",
            "        # Compute error\\n",
            "        target = np.zeros(len(probs))\\n",
            "        target[context_idx] = 1\\n",
            "        error = probs - target\\n",
            "        \\n",
            "        # Backprop (simplified)\\n",
            "        d_context = np.outer(h, error)\\n",
            "        d_embedding = np.dot(self.context_weights, error)\\n",
            "        \\n",
            "        # Update weights\\n",
            "        self.context_weights -= learning_rate * d_context\\n",
            "        self.embeddings[word_idx] -= learning_rate * d_embedding\\n",
            "        \\n",
            "        # Return loss\\n",
            "        return -np.log(probs[context_idx] + 1e-10)\\n",
            "\\n",
            "# Create and train model\\n",
            "model = SimpleWord2Vec(vocab_size=len(vocab), embedding_dim=10)\\n",
            "\\n",
            "print(\\"Training Word2Vec model...\\")\\n",
            "epochs = 100\\n",
            "losses = []\\n",
            "\\n",
            "for epoch in range(epochs):\\n",
            "    epoch_loss = 0\\n",
            "    for word, context in training_pairs:\\n",
            "        word_idx = vocab[word]\\n",
            "        context_idx = vocab[context]\\n",
            "        loss = model.train_step(word_idx, context_idx)\\n",
            "        epoch_loss += loss\\n",
            "    \\n",
            "    losses.append(epoch_loss / len(training_pairs))\\n",
            "    \\n",
            "    if (epoch + 1) % 20 == 0:\\n",
            "        print(f\\"Epoch {epoch+1}/{epochs}, Loss: {losses[-1]:.4f}\\")\\n",
            "\\n",
            "# Plot training loss\\n",
            "plt.figure(figsize=(10, 4))\\n",
            "plt.plot(losses)\\n",
            "plt.title('Word2Vec Training Loss')\\n",
            "plt.xlabel('Epoch')\\n",
            "plt.ylabel('Loss')\\n",
            "plt.grid(True)\\n",
            "plt.show()\\n",
            "\\n",
            "print(\\"\\\\n✅ Model trained!\\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Exploring Learned Embeddings\\n",
            "\\n",
            "Let's see what our model learned!"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def find_similar_words(word, vocab, embeddings, top_k=3):\\n",
            "    if word not in vocab:\\n",
            "        return []\\n",
            "    \\n",
            "    word_idx = vocab[word]\\n",
            "    word_vec = embeddings[word_idx]\\n",
            "    \\n",
            "    similarities = []\\n",
            "    for other_word, other_idx in vocab.items():\\n",
            "        if other_word != word:\\n",
            "            other_vec = embeddings[other_idx]\\n",
            "            sim = cosine_similarity(word_vec, other_vec)\\n",
            "            similarities.append((other_word, sim))\\n",
            "    \\n",
            "    # Sort by similarity\\n",
            "    similarities.sort(key=lambda x: x[1], reverse=True)\\n",
            "    return similarities[:top_k]\\n",
            "\\n",
            "# Test similarity\\n",
            "test_words = ['machine', 'learning', 'neural']\\n",
            "\\n",
            "for word in test_words:\\n",
            "    if word in vocab:\\n",
            "        similar = find_similar_words(word, vocab, model.embeddings, top_k=3)\\n",
            "        print(f\\"\\\\nWords similar to '{word}':\\")\\n",
            "        for sim_word, sim_score in similar:\\n",
            "            print(f\\"  {sim_word}: {sim_score:.3f}\\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Visualizing Embeddings\\n",
            "\\n",
            "Let's visualize our embeddings in 2D!"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.decomposition import PCA\\n",
            "\\n",
            "# Reduce to 2D using PCA\\n",
            "pca = PCA(n_components=2)\\n",
            "embeddings_2d = pca.fit_transform(model.embeddings)\\n",
            "\\n",
            "# Plot\\n",
            "plt.figure(figsize=(12, 8))\\n",
            "plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)\\n",
            "\\n",
            "# Label each point\\n",
            "for word, idx in vocab.items():\\n",
            "    x, y = embeddings_2d[idx]\\n",
            "    plt.annotate(word, (x, y), fontsize=10)\\n",
            "\\n",
            "plt.title('Word Embeddings Visualization (2D)')\\n",
            "plt.xlabel('PC1')\\n",
            "plt.ylabel('PC2')\\n",
            "plt.grid(True, alpha=0.3)\\n",
            "plt.show()\\n",
            "\\n",
            "print(\\"Words that appear in similar contexts should be closer together!\\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Word Analogies (The Magic of Embeddings!)\\n",
            "\\n",
            "Famous example: **king - man + woman ≈ queen**\\n",
            "\\n",
            "This works because embeddings capture relationships!"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def word_analogy(word_a, word_b, word_c, vocab, embeddings, top_k=3):\\n",
            "    \\\"\\\"\\\"\\n",
            "    Find word D such that: A is to B as C is to D\\n",
            "    E.g., 'king' - 'man' + 'woman' = ?\\n",
            "    \\\"\\\"\\\"\\n",
            "    if not all(w in vocab for w in [word_a, word_b, word_c]):\\n",
            "        return []\\n",
            "    \\n",
            "    # Get embeddings\\n",
            "    vec_a = embeddings[vocab[word_a]]\\n",
            "    vec_b = embeddings[vocab[word_b]]\\n",
            "    vec_c = embeddings[vocab[word_c]]\\n",
            "    \\n",
            "    # Compute target: b - a + c\\n",
            "    target = vec_b - vec_a + vec_c\\n",
            "    \\n",
            "    # Find closest word\\n",
            "    similarities = []\\n",
            "    for word, idx in vocab.items():\\n",
            "        if word not in [word_a, word_b, word_c]:\\n",
            "            vec = embeddings[idx]\\n",
            "            sim = cosine_similarity(target, vec)\\n",
            "            similarities.append((word, sim))\\n",
            "    \\n",
            "    similarities.sort(key=lambda x: x[1], reverse=True)\\n",
            "    return similarities[:top_k]\\n",
            "\\n",
            "# Try some analogies (with our small corpus)\\n",
            "print(\\"Attempting word analogies...\\")\\n",
            "print(\\"(Note: Our tiny corpus might not show perfect analogies)\\\\n\\")\\n",
            "\\n",
            "# Example: Try to find relationships\\n",
            "if 'machine' in vocab and 'deep' in vocab and 'learning' in vocab:\\n",
            "    result = word_analogy('machine', 'learning', 'deep', vocab, model.embeddings)\\n",
            "    print(\\"'machine' is to 'learning' as 'deep' is to:\\")\\n",
            "    for word, score in result:\\n",
            "        print(f\\"  {word}: {score:.3f}\\")\\n",
            "\\n",
            "print(\\"\\\\nWith more data and training, analogies become much more meaningful!\\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 10. Using Pre-trained Embeddings\\n",
            "\\n",
            "In practice, we use pre-trained embeddings:\\n",
            "- **Word2Vec** (Google): Trained on Google News\\n",
            "- **GloVe** (Stanford): Trained on Wikipedia + web data\\n",
            "- **FastText** (Facebook): Handles unknown words better\\n",
            "\\n",
            "Let's see how to use them!"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Example: Using gensim for pre-trained Word2Vec\\n",
            "print(\\"To use pre-trained embeddings, install gensim:\\")\\n",
            "print(\\"  pip install gensim\\\\n\\")\\n",
            "\\n",
            "print(\\"Then load pre-trained vectors:\\")\\n",
            "print(\\"\\\"\\\"\\")\\n",
            "print(\\"from gensim.models import KeyedVectors\\")\\n",
            "print(\\"\\")\\n",
            "print(\\"# Download from: https://code.google.com/archive/p/word2vec/\\")\\n",
            "print(\\"model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)\\")\\n",
            "print(\\"\\")\\n",
            "print(\\"# Find similar words\\")\\n",
            "print(\\"similar = model.most_similar('king', topn=5)\\")\\n",
            "print(\\"print(similar)\\")\\n",
            "print(\\"\\")\\n",
            "print(\\"# Word analogy\\")\\n",
            "print(\\"result = model.most_similar(positive=['woman', 'king'], negative=['man'])\\")\\n",
            "print(\\"print(result[0])  # Should be close to 'queen'!\\")\\n",
            "print(\\"\\\"\\\"\\")\\n",
            "\\n",
            "print(\\"\\\\n💡 Pre-trained embeddings save tons of training time!\\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Summary\\n",
            "\\n",
            "### What We Learned:\\n",
            "\\n",
            "1. **Tokenization**: Breaking text into words/subwords\\n",
            "2. **Vocabulary**: Mapping words to numbers\\n",
            "3. **One-hot encoding**: Simple but no semantic meaning\\n",
            "4. **Word embeddings**: Dense vectors that capture meaning\\n",
            "5. **Word2Vec**: Learning embeddings from context\\n",
            "6. **Skip-gram**: Predicting context from word\\n",
            "7. **Analogies**: Embeddings capture relationships!\\n",
            "\\n",
            "### Key Insights:\\n",
            "- Similar words → similar vectors\\n",
            "- Vector arithmetic captures meaning\\n",
            "- Context is everything!\\n",
            "\\n",
            "### Next Steps:\\n",
            "👉 **Lesson 2**: Build a sentiment analyzer using embeddings\\n",
            "👉 **Lesson 3**: Learn about attention mechanism\\n",
            "\\n",
            "---\\n",
            "\\n",
            "## Practice Exercises\\n",
            "\\n",
            "1. **Expand the corpus**: Add more sentences and retrain\\n",
            "2. **Try larger embeddings**: Use 50 or 100 dimensions\\n",
            "3. **Implement CBOW**: Predict word from context (opposite of skip-gram)\\n",
            "4. **Download GloVe**: Use pre-trained embeddings for your own text\\n",
            "\\n",
            "Great work! You now understand how computers represent text! 🚀"
        ]
    }
]

print("="*60)
print("Creating Phase 2, Lesson 1...")
print("="*60)
create_notebook_cells("phase2_text_and_embeddings", "01", "text_processing_and_embeddings", phase2_lesson1_cells)
print("\n✅ Phase 2, Lesson 1 complete!\n")
