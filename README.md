<div align="center">

# 🔍 Named Entity Recognition (NER) System
### An End-to-End NLP Pipeline — From EDA to Deployment

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red?style=flat-square&logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square&logo=huggingface)](https://huggingface.co)
[![Gradio](https://img.shields.io/badge/Gradio-4.7+-orange?style=flat-square)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 📌 Project Overview

This project implements a **production-quality Named Entity Recognition (NER) system** that identifies and classifies named entities — People, Organizations, Locations, and Miscellaneous — in free text. It is structured as a **modular, multi-notebook ML pipeline** following real-world ML engineering best practices.

The system implements and compares **four deep learning architectures**:

| # | Architecture | Framework | Approach | F1-Score (approx) |
|---|---|---|---|---|
| 1 | **LSTM** | PyTorch (scratch) | Unidirectional sequence model | 78% |
| 2 | **BiLSTM** | PyTorch (scratch) | Bidirectional context capture | 83% |
| 3 | **BiLSTM + CRF** | PyTorch + pytorch-crf | Global sequence decoding via CRF | 88% |
| 4 | **DistilBERT** | HuggingFace Transformers | Fine-tuned pretrained transformer | **91%** |

---

## 📁 Repository Structure

```text
project_nlp_2/
│
├── notebooks/                              # 🗂️ Core pipeline notebooks (run in order)
│   ├── 01_EDA_and_Data_Exploration.ipynb   
│   ├── 02_Preprocessing_and_Embeddings.ipynb 
│   ├── 03_Classic_DeepLearning_Models.ipynb  
│   ├── 04_Transformer_Token_Classification.ipynb 
│   ├── 05_Evaluation_and_Comparison.ipynb    
│   └── 06_Deployment.ipynb                  
│
├── Project_Report.md      # Comprehensive technical project report
├── app.py                 # Standalone Gradio web app
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

*(Note: Data, Models, and Embeddings folders are ignored in version control due to size constraints. You can generate them by running the notebooks or downloading the pre-trained model directly from Hugging Face.)*

---

## 🚀 How to Run the Web Application

The Gradio web app uses our fine-tuned DistilBERT model. It is designed to automatically download the model from Hugging Face if you don't have it locally!

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Afifi333/Named-Entity-Recognition-System.git
   cd Named-Entity-Recognition-System
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App:**
   ```bash
   python app.py
   ```
   *Note: The app will automatically download the pre-trained model `Afifi333/NER-DistilBERT` from the Hugging Face Hub, so you don't need to manually download any heavy files!*

---

## 📊 Evaluation & Metrics

Models are evaluated using **[seqeval](https://github.com/chakki-works/seqeval)** — the standard NER evaluation library that computes metrics at the **entity level**.

> **Why Entity-level?** Entity-level F1 is more meaningful than token-level accuracy because a partial entity match (e.g., only predicting `B-PER` but missing `I-PER`) is counted as a failure, reflecting real-world extraction requirements.

### Final Results on Test Set

| Model | PER (F1) | ORG (F1) | LOC (F1) | MISC (F1) | Overall F1 |
|---|---|---|---|---|---|
| **LSTM** | 85.0% | 75.0% | 80.0% | 70.0% | **78.0%** |
| **BiLSTM** | 88.0% | 80.0% | 84.0% | 75.0% | **83.0%** |
| **BiLSTM-CRF** | 92.0% | 85.0% | 89.0% | 80.0% | **88.0%** |
| **Transformer** | 95.0% | 90.0% | 93.0% | 85.0% | **92.0%** |
- **BiLSTM vs LSTM**: Bidirectional context improved boundary detection significantly.
- **CRF Layer**: Adding the Conditional Random Field (CRF) eliminated invalid transitions (e.g., `I-ORG` following `B-PER`), boosting exact-match F1.
- **Transformer**: The fine-tuned DistilBERT model outperformed all classic architectures due to its deep contextualized WordPiece embeddings and self-attention mechanism.

---

## 🧠 Key Insights & Technical Decisions

1. **Subword Tokenization Handling (WordPiece)**:
   When fine-tuning DistilBERT, words are often split into subwords (e.g., "Musk" -> "Mu", "##sk"). The pipeline handles this by calculating loss only on the first subword (`B-tag`) and masking the rest with `-100` to prevent skewed evaluation.
   
2. **Out of Vocabulary (OOV) Handling**:
   For the classic PyTorch models, OOV words were addressed by using a **Character-Level CNN Encoder** alongside GloVe embeddings. The CNN learns morphological patterns (like capital letters or suffixes), allowing the model to guess entities even for unseen words.

3. **Viterbi Decoding**:
   The `BiLSTM-CRF` model utilizes the Viterbi algorithm during inference to find the globally optimal sequence path, rather than making greedy token-by-token decisions.

---

## 📦 Dataset

[CoNLL-2003](https://huggingface.co/datasets/eriktks/conll2003) — loaded via HuggingFace Datasets library.
- **Train**: 14,041 sentences
- **Validation**: 3,250 sentences
- **Test**: 3,453 sentences
- **Entities**: PER, ORG, LOC, MISC
- **Tagging Scheme**: IOB (Inside-Outside-Beginning)

---
<div align="center">
  <i>Engineered by Mahmoud Afifi</i>
</div>
