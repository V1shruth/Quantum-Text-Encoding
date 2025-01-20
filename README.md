# Quantum-Text-Encoding

## Overview

The Quantum Text Encoding project combines classical machine learning and quantum computing concepts to encode text into quantum circuits. This project explores the synergy between advanced natural language processing (NLP) techniques and quantum computing, paving the way for innovative solutions in text analysis and encoding.

The workflow utilizes transformer-based models for text embedding, dimensionality reduction for compatibility with quantum systems, and amplitude encoding to transform classical data into quantum states.

## Features

1. **Transformer-Based Embeddings**: Extracts high-quality text embeddings using pre-trained Hugging Face models (e.g., DistilBERT).
2. **Dimensionality Reduction**: Reduces high-dimensional embeddings using PCA for efficient quantum state encoding.
3. **Quantum Encoding**:
   - Implements amplitude encoding to prepare quantum states.
   - Applies sinusoidal positional encoding for transformer-like properties.
   - Builds hierarchical quantum circuits for encoding grouped data.
4. **Parametric Entanglement**: Utilizes CRZ gates for learning-based entanglement of quantum states.

---

## Project Structure

- `TransformerEmbedder`: Handles text tokenization and embedding extraction using Hugging Face models.
- `DimensionalityReducer`: Performs dimensionality reduction with PCA to prepare embeddings for quantum encoding.
- `QuantumEncoder`: Encodes classical embeddings into quantum states with optional positional encoding.
- `ParametricEntangler`: Adds entanglement between qubits in quantum circuits.
- `HierarchicalQuantumEncoder`: Combines multiple quantum circuits into hierarchical structures.
- `demo_transformer_based_encoding`: Integrates all components into an end-to-end workflow for text encoding.

---

## Dependencies

The project relies on the following Python libraries:

### Core Dependencies

- **Python**: Version 3.10 (recommended).
- **NumPy**: For numerical operations.
- **scikit-learn**: For PCA-based dimensionality reduction.

### Natural Language Processing

- **nltk**: Tokenization support.
- **transformers**: Pre-trained Hugging Face models for text embeddings.

### Quantum Computing

- **Qiskit**: For building and simulating quantum circuits.
- **FakeAthens** (optional): Mock backend for quantum simulations.

### Miscellaneous

- **logging**: For debug and status messages.
- **warnings**: To suppress runtime and deprecation warnings.

---

## Installation

Follow these steps to set up the environment and install the required dependencies:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/V1shruth/Quantum-Text-Encoding.git
   cd Quantum-Text-Encoding
   ```

2. **Set Up Environment**:
   Use Conda to create a virtual environment:
   ```bash
   conda create --name qte_py310 python=3.10 -y
   conda activate qte_py310
   ```

3. **Install Dependencies**:
   Install the required libraries:
   ```bash
   pip install numpy scikit-learn nltk transformers qiskit
   ```

4. **Download NLTK Tokenizer Data**:
   Run the following in your Python environment:
   ```python
   import nltk
   nltk.download('punkt', quiet=True)
   ```

---

## Usage

### End-to-End Workflow
The main function `demo_transformer_based_encoding` integrates all components. Below is an example usage:

```python
sample_texts = [
    "Quantum computing in modern machine learning research",
    "Artificial intelligence uses deep neural networks",
    "Data science and big data analytics revolution"
]

for txt in sample_texts:
    print("=======================================================")
    print(f"Encoding text: {txt}")
    distribution = demo_transformer_based_encoding(
        text=txt,
        hf_model_name="distilbert-base-uncased",
        pca_power=2,    # => dimension=4 after PCA
        n_qubits=2,     # => amplitude-encode 4D vectors
        group_size=2,
        multi_level=True
    )
    print(f"Final distribution (size={len(distribution)}):\n{distribution}\n")
```

### Steps in Encoding
1. **Tokenization**: Splits text into subword tokens.
2. **Embedding Extraction**: Converts text tokens into high-dimensional vectors.
3. **Dimensionality Reduction**: Reduces embedding dimensions for quantum encoding.
4. **Quantum Circuit Building**:
   - Constructs quantum circuits using amplitude encoding and sinusoidal positional encoding.
   - Adds entanglement between qubits.
5. **Hierarchical Encoding** (optional): Groups data for multi-level circuit encoding.

---

## Key Classes and Functions

### `TransformerEmbedder`
- **Purpose**: Tokenizes text and extracts embeddings using Hugging Face transformers.
- **Methods**:
  - `tokenize(text: str)`: Tokenizes input text.
  - `get_token_embeddings(text: str)`: Returns token embeddings.

### `DimensionalityReducer`
- **Purpose**: Reduces embedding dimensions using PCA.
- **Methods**:
  - `fit_transform(embeddings)`: Fits PCA and reduces dimensions.

### `QuantumEncoder`
- **Purpose**: Encodes vectors into quantum circuits.
- **Methods**:
  - `amplitude_encode(vector)`: Prepares amplitude-encoded quantum state.
  - `sinusoidal_position_encoding(qc, position)`: Adds positional encoding to circuit.

### `ParametricEntangler`
- **Purpose**: Adds entanglement using parametric CRZ gates.
- **Methods**:
  - `apply(qc, param_values=None)`: Applies entanglement to quantum circuit.

### `HierarchicalQuantumEncoder`
- **Purpose**: Encodes groups of data hierarchically into quantum circuits.
- **Methods**:
  - `build_level_circuit(vectors, positions)`: Builds group circuits with entanglement.

---

## Future Improvements

1. **Scalability**: Optimize quantum state preparation for larger text inputs.
2. **Backend Integration**: Extend support for real quantum hardware via IBMQ.
3. **Enhanced Models**: Experiment with advanced transformer models like BERT-large or GPT.
4. **Visualization**: Add tools for visualizing quantum circuits and encoded states.

