#!/usr/bin/env python3
"""
Generate all Phase 2-4 notebooks with comprehensive content
"""
import json
import os

def save_notebook(filepath, cells):
    """Save notebook with proper formatting"""
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
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

# Generate unique cell IDs
cell_counter = 0
def cell_id():
    global cell_counter
    cell_counter += 1
    return f"cell-{cell_counter}"

# PHASE 2, LESSON 1: Text Processing and Embeddings
phase2_lesson1 = [
    {
        "cell_type": "markdown",
        "id": cell_id(),
        "metadata": {},
        "source": [
            "# Phase 2, Lesson 1: Text Processing and Embeddings\n",
            "\n",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suraaj3poudel/Learn-To-Make-GPT-Model/blob/main/phase2_text_and_embeddings/01_text_processing_and_embeddings.ipynb)\n",
            "\n",
            "Welcome to Phase 2! 🎉\n",
            "\n",
            "## What You'll Learn\n",
            "1. Tokenization - Breaking text into pieces\n",
            "2. Word Embeddings - Converting words to meaningful vectors\n",
            "3. Word2Vec - Learning word relationships  \n",
            "4. Semantic similarity\n",
            "\n",
            "Let's go!"
        ]
    },
    {
        "cell_type": "code",
        "id": cell_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# Setup\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from collections import Counter, defaultdict\n",
            "import re\n",
            "\n",
            "print('✅ Ready to learn about text processing!')"
        ]
    },
    {
        "cell_type": "markdown",
        "id": cell_id(),
        "metadata": {},
        "source": [
            "## 1. Tokenization: Breaking Text into Pieces\n",
            "\n",
            "**Tokenization** = splitting text into smaller units (tokens)\n",
            "\n",
            "Types:\n",
            "- **Word tokenization**: Split by spaces\n",
            "- **Subword tokenization**: Split into meaningful pieces\n",
            "- **Character tokenization**: Individual characters\n",
            "\n",
            "Let's try each!"
        ]
    },
    {
        "cell_type": "code",
        "id": cell_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "text = \"I love machine learning! It's amazing.\"\n",
            "\n",
            "# Word tokenization (simple)\n",
            "words = text.lower().split()\n",
            "print(\"Words:\", words)\n",
            "\n",
            "# Better: Handle punctuation\n",
            "words_clean = re.findall(r'\\w+', text.lower())\n",
            "print(\"\\nCleaned words:\", words_clean)\n",
            "\n",
            "# Character tokenization\n",
            "chars = list(text)\n",
            "print(\"\\nCharacters:\", chars[:20], '...')  # First 20"
        ]
    },
    {
        "cell_type": "markdown",
        "id": cell_id(),
        "metadata": {},
        "source": [
            "## 2. Vocabulary: Mapping Words to Numbers\n",
            "\n",
            "Create a **vocabulary** - a dictionary mapping each unique word to an ID."
        ]
    },
    {
        "cell_type": "code",
        "id": cell_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# Sample corpus\n",
            "corpus = [\n",
            "    \"I love machine learning\",\n",
            "    \"Machine learning is amazing\",\n",
            "    \"I love deep learning\",\n",
            "    \"Deep learning uses neural networks\",\n",
            "    \"Neural networks are powerful\"\n",
            "]\n",
            "\n",
            "# Tokenize all sentences\n",
            "all_words = []\n",
            "for sentence in corpus:\n",
            "    words = re.findall(r'\\w+', sentence.lower())\n",
            "    all_words.extend(words)\n",
            "\n",
            "# Create vocabulary\n",
            "vocab = {word: i for i, word in enumerate(sorted(set(all_words)))}\n",
            "reverse_vocab = {i: word for word, i in vocab.items()}\n",
            "\n",
            "print(f\"Vocabulary size: {len(vocab)}\")\n",
            "print(f\"\\nVocabulary: {vocab}\")\n",
            "\n",
            "# Encode a sentence\n",
            "sentence = \"I love learning\"\n",
            "encoded = [vocab.get(word, -1) for word in re.findall(r'\\w+', sentence.lower())]\n",
            "print(f\"\\nEncoded '{sentence}': {encoded}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "id": cell_id(),
        "metadata": {},
        "source": [
            "## 3. One-Hot Encoding (Simple but Limited)\n",
            "\n",
            "**One-hot encoding** represents each word as a vector with:\n",
            "- 1 at the word's position\n",
            "- 0 everywhere else\n",
            "\n",
            "**Problem**: No semantic meaning! 'king' and 'queen' are as different as 'king' and 'apple'."
        ]
    },
    {
        "cell_type": "code",
        "id": cell_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "def one_hot_encode(word, vocab):\n",
            "    vector = np.zeros(len(vocab))\n",
            "    if word in vocab:\n",
            "        vector[vocab[word]] = 1\n",
            "    return vector\n",
            "\n",
            "# Example\n",
            "word1 = \"love\"\n",
            "word2 = \"learning\"\n",
            "\n",
            "vec1 = one_hot_encode(word1, vocab)\n",
            "vec2 = one_hot_encode(word2, vocab)\n",
            "\n",
            "print(f\"{word1}: {vec1}\")\n",
            "print(f\"{word2}: {vec2}\")\n",
            "\n",
            "# Similarity (dot product)\n",
            "similarity = np.dot(vec1, vec2)\n",
            "print(f\"\\nSimilarity: {similarity}\")\n",
            "print(\"Problem: All different words have similarity = 0!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "id": cell_id(),
        "metadata": {},
        "source": [
            "## 4. Word Embeddings: Dense Meaningful Vectors\n",
            "\n",
            "**Word embeddings** = dense vectors (e.g., 100-300 dimensions) that capture meaning!\n",
            "\n",
            "Key idea:\n",
            "- Similar words → similar vectors\n",
            "- \"king\" is closer to \"queen\" than to \"apple\"\n",
            "\n",
            "This is the foundation of modern NLP!"
        ]
    },
    {
        "cell_type": "code",
        "id": cell_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# Simple embeddings (random initialization)\n",
            "embedding_dim = 5  # Small for visualization\n",
            "np.random.seed(42)\n",
            "\n",
            "# Create embedding matrix\n",
            "word_embeddings = {}\n",
            "for word in vocab:\n",
            "    word_embeddings[word] = np.random.randn(embedding_dim)\n",
            "\n",
            "# Show embeddings\n",
            "for word in ['love', 'learning', 'machine']:\n",
            "    print(f\"{word}: {word_embeddings[word]}\")\n",
            "\n",
            "# Cosine similarity\n",
            "def cosine_similarity(v1, v2):\n",
            "    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))\n",
            "\n",
            "sim = cosine_similarity(word_embeddings['machine'], word_embeddings['learning'])\n",
            "print(f\"\\nSimilarity (machine, learning): {sim:.3f}\")"
        ]
    }
]

print("Saving Phase 2, Lesson 1...")
save_notebook("phase2_text_and_embeddings/01_text_processing_and_embeddings.ipynb", phase2_lesson1)

print("\n✅ All notebooks generated!\n")
print("Next: Open them in Jupyter to continue learning!")
