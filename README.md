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

| # | Architecture | Framework | Approach |
|---|---|---|---|
| 1 | **LSTM** | PyTorch (scratch) | Unidirectional sequence model |
| 2 | **BiLSTM** | PyTorch (scratch) | Bidirectional context capture |
| 3 | **BiLSTM + CRF** | PyTorch + pytorch-crf | Global sequence decoding via CRF |
| 4 | **DistilBERT** | HuggingFace Transformers | Fine-tuned pretrained transformer |

---

## 📁 Repository Structure

```text
project_nlp_2/
│
├── notebooks/                              # 🗂️ Core pipeline notebooks (run in order)
│   ├── 01_EDA_and_Data_Exploration.ipynb   # Dataset analysis & IOB tagging visualization
│   ├── 02_Preprocessing_and_Embeddings.ipynb # GloVe loading, OOV, DataLoaders
│   ├── 03_Classic_DeepLearning_Models.ipynb  # LSTM, BiLSTM, BiLSTM-CRF training
│   ├── 04_Transformer_Token_Classification.ipynb # DistilBERT fine-tuning
│   ├── 05_Evaluation_and_Comparison.ipynb    # seqeval metrics, comparison charts
│   └── 06_Deployment.ipynb                  # Gradio web app deployment
│
├── data/                  # Saved vocabularies and DataLoader objects
├── models/                # Trained model checkpoints (.pt files + HuggingFace model dir)
├── embeddings/            # GloVe / FastText embedding files
├── outputs/               # Final metric reports (JSON, CSV)
├── figures/               # Generated plots and visualizations
├── logs/                  # Training logs (JSON)
├── deployment/            # Standalone Gradio app (app.py + requirements.txt)
├── utils/                 # (Utility scripts — referenced inside notebooks)
│
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

---

## 🔗 Pipeline Flow

```
01_EDA_and_Data_Exploration
           │
           ▼
02_Preprocessing_and_Embeddings
           │
   ┌───────┴───────┐
   │               │
   ▼               ▼
03_Classic_Models  04_Transformer
(LSTM/BiLSTM/CRF)  (DistilBERT)
   │               │
   └───────┬───────┘
           │
           ▼
05_Evaluation_and_Comparison
           │
           ▼
06_Deployment (Gradio)
```

---

## 🚀 How to Run

### ▶️ Option A: Kaggle (Recommended — Free GPU)
1. Go to [Kaggle](https://kaggle.com) → **New Notebook**
2. Click **File → Import Notebook** and upload notebooks one by one (01 → 06)
3. Enable **GPU Accelerator** (Settings → Accelerator → GPU T4)
4. Run cells sequentially from top to bottom

### 💻 Option B: Local Machine
```bash
# 1. Clone this repository
git clone https://github.com/yourusername/project-nlp-2.git
cd project-nlp-2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download GloVe embeddings (100d)
# Place glove.6B.100d.txt inside embeddings/

# 4. Open Jupyter
jupyter notebook notebooks/

# 5. Run notebooks in order: 01 → 02 → 03 → 04 → 05 → 06
```

### 🌐 Option C: Run Gradio App Standalone
```bash
python deployment/app.py
```

---

## 📊 Evaluation

Models are evaluated using **[seqeval](https://github.com/chakki-works/seqeval)** — the standard NER evaluation library that computes metrics at the **entity level** (not token level), ensuring accurate Precision, Recall, and F1-score for each entity type.

> Entity-level F1 is more meaningful than token-level accuracy because a partial entity match (e.g., only `B-PER` detected, not `I-PER`) is counted as a failure.

---

## 🧠 Key Insights

**Why BiLSTM-CRF outperforms BiLSTM:**
- BiLSTM assigns tags independently per token with no awareness of neighbors
- The CRF layer learns a **transition matrix** `T[i][j]` = score of going from tag `i` → tag `j`
- This prevents invalid sequences like `I-ORG` following `B-PER`
- The Viterbi algorithm finds the **globally optimal** tag sequence, improving boundary detection

---

## 📦 Dataset

[CoNLL-2003](https://huggingface.co/datasets/eriktks/conll2003) — loaded via HuggingFace Datasets library.
- **Train**: 14,041 sentences
- **Validation**: 3,250 sentences
- **Test**: 3,453 sentences
- **Entities**: PER, ORG, LOC, MISC
- **Tagging Scheme**: IOB (Inside-Outside-Beginning)
