# TDSE_LLM_Text_Preprocessing_Foundations

**Created by**

Juan Carlos Leal Cruz

## **Project Overview**
Hands-on exploration of text preprocessing, tokenization fundamentals, and embedding generation for Large Language Models. This project is based on Chapter 2 of *"Build a Large Language Model (From Scratch)"* by Sebastian Raschka.

This project demonstrates the essential preprocessing pipeline required to transform raw text into mathematical representations that neural networks can process. It covers tokenization strategies, embedding generation, and data sampling techniques that form the foundation of modern language models and agentic AI systems.

## **Prerequisites**

To run this laboratory you will need:

- Python 3.8 or higher (As a recomendation, the newer versions are better due to the compatibility with the creations of virtual environments in multiple IDEs)
- Jupyter Notebook or JupyterLab
- The following Python libraries:
  - `numpy`
  - `matplotlib`
  - `tiktoken`
  - `torch`

You can add the previous libraries by using the following command:
```
pip install torch tiktoken numpy matplotlib
```
 
## Execution
To run this laboratory, follow the steps below:

1. Clone the repository and navigate to the folder:
   ```
   git clone <Repository_URL>
   cd <Repository_name>
   ```

2. Setup the virtual environment
   ```
   python -m venv venv
   source venv/bin/activate       # On Linux/Mac
   venv\Scripts\activate          # On Windows
   ```

3. Start running each block of code so you can see the results

___

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

## **Useful Concepts**
### **Tokenization**
Tokenization bridges human language and machine learning. While naive word-level splitting results in massive, inefficient vocabularies, modern models use Byte Pair Encoding (BPE). BPE efficiently handles rare words by breaking them into subword units, ensuring there is never an "unknown token" while keeping the vocabulary size manageable.

### **Embeddings**
Embeddings are dense vector representations that capture semantic meaning and geometric relationships. Because the Transformer architecture is naturally permutation-invariant, Positional Embeddings are added to the token vectors so the model can understand causality and the order of operations—a critical feature for agentic systems.

### **Sliding Window**
LLMs learn via self-supervised next-token prediction using overlapping context windows. Each sample teaches the model to predict token N+1 given tokens 1 to N.

## **Key Experiments Results**
The `embeddings.ipynb` notebook includes a specific experiment testing the relationship between sliding window parameters and dataset generation.

### **Hypothesis Tested:**
By decreasing the `stride` (increasing the overlap between chunks), we can linearly increase the size of our training dataset without requiring additional raw text. We evaluated:
1. Inverse Relationship: The number of training samples will be inversely proportional to the stride.
2. Data Augmentation: A stride of 1 will result in the maximum possible data efficiency, effectively "augmenting" the dataset by forcing the model to learn the same token's meaning across every possible position in the context window.

### **Experimental Results**
- **Context Length = 4**
  - `stride=4` (0% overlap): **1,286 samples** | Efficiency: **1.00x**
  - `stride=2` (50% overlap): **2,571 samples** | Efficiency: **2.00x**
  - `stride=1` (75% overlap): **5,141 samples** | Efficiency: **4.00x**

- **Context Length = 8**
  - `stride=8` (0% overlap): **643 samples** | Efficiency: **1.00x**
  - `stride=4` (50% overlap): **1,285 samples** | Efficiency: **2.00x**

**Conclusion:**
The results strongly confirm the hypothesis.

     


