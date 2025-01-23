# Reproducing: Learned in Translation: Contextualized Word Vectors
Reproducing [Learned in Translation: Contextualized Word Vectors](https://arxiv.org/abs/1708.00107) for the class of Deep Learning for Natural Language Processing of TU Wien semester W2024. The authors use a deep LSTM encoder from an attentional sequence-to-sequence model trained for machine translation (MT) to contextualize word vectors. Their results show that adding these context vectors (CoVe) improves performance over using only unsupervised word and character vectors on a wide variety of common NLP tasks

# GLOVE embeddings

# Neural Machine Translation (NMT) model
Feed GLOVE representations to Encoder-Decoder architecture with global attention module

Obtain COVE

# The Biattentive Classification Network (BCN)

Input [GLOVE;COVE]
Output Probability distribution over possible classes

# Dynamic Coattention Network (DCN)