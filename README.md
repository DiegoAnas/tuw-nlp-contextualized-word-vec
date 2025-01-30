
# **192.039 Deep Learning for Natural Language Processing 2024W - Final Project**

## **Overview**

Reproducing [Learned in Translation: Contextualized Word Vectors](https://arxiv.org/abs/1708.00107) for the class of Deep Learning for Natural Language Processing of TU Wien semester W2024. The authors use a deep LSTM encoder from an attentional sequence-to-sequence model trained for machine translation (MT) to contextualize word vectors. Their results show that adding these context vectors (CoVe) improves performance over using only unsupervised word and character vectors on a wide variety of common NLP tasks.

### Translation
Encoder-Decoder Architecture using Global Attention module for the decoder.

### Tasks
To evaluate the performance of this transfer learning method they utilize two architectures:

#### Biattention Classification Network
This model uses Biattention mechanism to classification tasks from IMDB, SST, TREC and SNLI datasets.

#### Dynamic Coattention Networks 
This model from the paper Dynamic Coattention Networks For Question Answering is used to generate answers for the SQuaD challenge.

### Extensions
This work is an extension of the paper _Learned in Translation: Contextualized Word Vectors_ with improvements using **BERT-based embeddings**.
⚠ **Note:** The BERT-based embedding integration is still a **work in progress** and may not be fully functional yet. Further debugging and testing are required to ensure stability.

## **Installation**
### **Requirements**
Ensure you have the necessary dependencies installed:
```bash
pip install torch transformers BERTembedding datasets
```

## **Next Steps**
- Debug and stabilize BERT-based embeddings in BCN.
- Experiment with different BERT variants (`bert-large`, `roberta-base`).
- Fine-tune BERT layers for **domain-specific tasks**.



