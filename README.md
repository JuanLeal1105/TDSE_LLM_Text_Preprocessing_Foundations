# TDSE_LLM_Text_Preprocessing_Foundations

**Created by**

Juan Carlos Leal Cruz

## **Project Overview**
Hands-on exploration of text preprocessing, tokenization fundamentals, and embedding generation for Large Language Models. This project is based on Chapter 2 of *"Build a Large Language Model (From Scratch)"* by Sebastian Raschka.

This project demonstrates the essential preprocessing pipeline required to transform raw text into mathematical representations that neural networks can process. It covers tokenization strategies, embedding generation, and data sampling techniques that form the foundation of modern language models and agentic AI systems.

## **Notebooks (What you'll find in the lab)**
1. `ch2.ipynb` - Working with Text Data

   The original chapter notebook containing the foundational code examples, visual illustrations of embedding spaces, and the author's reference implementation for text data preparation.

2. `embeddings.ipynb` - Text Preprocessing Foundations

   A custom, cell-by-cell implementation and analysis exploring:
   - Tokenization Strategies:
     - Easeline implementation of a simple word-level tokenizer.
     - Transition to Byte Pair Encoding (BPE) using OpenAI's tiktoken.
     - Analysis of how BPE handles rare words and subword semantics.
   - Training with Sliding Window:
     - Implementation of a PyTorch `Dataset` and `DataLoader`.
     - Context window management and next-token prediction setup.
   - VectorEmbeddings:
     - Initializing and `applying torch.nn.Embedding` for tokens.
     - Implementing and adding Absolute Positional Embeddings to maintain sequence order.
     - Exploring the mathematical relationship between lookup tables and Neural Network layers.
   - Hyperparameter Experimentation:
     A detailed experiment analyzing the trade-off between computational cost and data augmentation by altering the `stride` parameter.


     


