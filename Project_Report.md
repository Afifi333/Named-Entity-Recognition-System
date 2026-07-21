# 📄 Technical Project Report: Named Entity Recognition (NER) System

**Author:** Mahmoud Afifi  
**Date:** July 2026  
**Domain:** Natural Language Processing (NLP)

---

## 1. Executive Summary

This project presents an end-to-end Machine Learning pipeline for Named Entity Recognition (NER). The objective was to build and evaluate multiple architectures — from fundamental Sequence Models (LSTM, BiLSTM), to probabilistic graphical models (CRF), to state-of-the-art self-attention mechanisms (Transformers) — in order to extract four distinct entities (PER, ORG, LOC, MISC) from raw text.

The final deliverable is a fully functional Gradio web application powered by a fine-tuned DistilBERT model, achieving an entity-level F1 score of ~91%.

## 2. Methodology & Architecture

The project was divided into a professional 6-stage pipeline:

1. **Exploratory Data Analysis (EDA):** Analyzed the CoNLL-2003 dataset, visualizing class imbalances and IOB (Inside-Outside-Beginning) tagging distributions.
2. **Preprocessing & Embeddings:** Handled Out-of-Vocabulary (OOV) tokens by implementing a Character-Level CNN encoder, combined with 100-dimensional GloVe word embeddings.
3. **Classic Deep Learning:** Built PyTorch models from scratch. 
   - **LSTM:** Baseline unidirectional context.
   - **BiLSTM:** Bidirectional context, drastically improving recall.
   - **BiLSTM-CRF:** Added a Conditional Random Field layer and Viterbi decoding to learn transition probabilities (e.g., forbidding `I-ORG` from following `B-PER`), eliminating boundary errors.
4. **Transformer Fine-tuning:** Leveraged HuggingFace to fine-tune DistilBERT. Handled WordPiece subword tokenization by utilizing `-100` attention masks to ignore non-leading subwords during loss calculation.
5. **Evaluation:** Conducted strict entity-level evaluation using `seqeval`.
6. **Deployment:** Wrapped the HuggingFace pipeline in a user-friendly Gradio interface.

## 3. Results & Evaluation

Evaluation was performed using the strict **entity-level F1-score**, which requires exact matches for both the boundary and the class of the entity.

| Model | Precision | Recall | F1-Score | Parameter Count |
|---|---|---|---|---|
| LSTM | ~0.76 | ~0.80 | **0.78** | ~920K |
| BiLSTM | ~0.81 | ~0.85 | **0.83** | ~1.8M |
| BiLSTM + CRF | ~0.89 | ~0.87 | **0.88** | ~1.8M |
| DistilBERT | ~0.90 | ~0.92 | **0.91** | ~66M |

**Key Finding:** The CRF layer contributed to a massive ~5% jump in F1-score over the standard BiLSTM by enforcing strict transition rules, proving the value of structured prediction in sequence labeling. However, the pre-trained contextual knowledge of DistilBERT ultimately yielded the best overall performance.

## 4. Engineering Best Practices

- **Modularity:** Code was split into logical, focused notebooks (01 to 06).
- **Reproducibility:** Global random seeds were strictly enforced across PyTorch, NumPy, and random libraries.
- **Scalability:** The Gradio application was designed with fallback mechanisms to automatically fetch the model weights from the Hugging Face Hub if local files are missing, keeping the GitHub repository lightweight.

## 5. Conclusion

The project successfully demonstrates a deep understanding of NLP evolution, moving from classic word embeddings and recurrent neural networks to modern transformer architectures. The final deployed application operates accurately in real-time, fulfilling all project requirements.
