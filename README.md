# SentimentScope – IMDB Sentiment Classification

A PyTorch-based Transformer model for classifying IMDB movie reviews as **Positive** or **Negative**.

## Project Overview

SentimentScope is a deep learning project that uses a custom Transformer architecture to perform sentiment analysis on movie reviews.

The model is trained from scratch using the IMDB dataset, with the `bert-base-uncased` tokenizer used for text tokenization.

**Objective:** Build a sentiment classification model that achieves more than 75% test accuracy.

## Dataset

- **Dataset:** IMDB Movie Reviews
- **Training samples:** 25,000
- **Testing samples:** 25,000
- **Classes:** Positive (1) and Negative (0)
- **Training split:** 90% training and 10% validation
- **Maximum sequence length:** 128 tokens

The dataset can be downloaded from the [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

## Model Architecture

The project implements a custom Transformer-based classification model using PyTorch.

Key components:
- Token Embedding
- Positional Embedding
- Multi-Head Self-Attention
- Feed-Forward Neural Network
- Layer Normalization
- Classification Head

**Tokenizer:** `bert-base-uncased`  
**Framework:** PyTorch  
**Model:** Custom Transformer (trained from scratch)

## Experiments and Results

Two model configurations were evaluated.

| Parameter | EXP 1 | EXP 2 |
|---|---:|---:|
| Embedding Dimension | 128 | 256 |
| Transformer Layers | 4 | 4 |
| Attention Heads | 4 | 8 |
| Training Epochs | 5 | 5 |
| Learning Rate | 0.0003 | 0.0002 |
| Best Validation Accuracy | 79.60% | **79.64%** |

EXP 2 achieved the highest validation accuracy and was selected as the final model.

## Repository Contents

| File | Description |
|---|---|
| `SentimentScope_starter.ipynb` | Complete implementation, data preparation, training, evaluation, and inference |
| `sentimentscope_final_model.pth` | Trained final model checkpoint (EXP 2) |
| `SentimentScope_Simple_Report.pdf` | Project report with experiment results and key takeaways |

## Requirements

Install the required Python packages:

```bash
pip install torch transformers pandas numpy matplotlib scikit-learn
```

## How to Run

1. Clone this repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

2. Navigate to the project directory:

```bash
cd <YOUR_REPOSITORY_NAME>
```

3. Install the required packages.

4. Open `SentimentScope_starter.ipynb` using Jupyter Notebook or JupyterLab.

5. Download and extract the IMDB dataset.

6. Execute the notebook cells in order to explore the dataset, train the model, and evaluate its performance.

The tokenizer will be downloaded automatically when initialized for the first time.

## Model Checkpoint

The final trained model is available as:

`sentimentscope_final_model.pth`

The checkpoint contains the trained model weights, model configuration, maximum sequence length, and experiment metadata.

## Key Takeaways

- Successfully implemented a custom Transformer architecture for binary sentiment classification.
- Increasing the embedding dimension and attention heads resulted in a small improvement in validation accuracy.
- Saved the best-performing model checkpoint for future inference and reuse.

## Author

**Thillairam**

Developed as part of the SentimentScope – IMDB Sentiment Classification project.
