import torch
from BERTembedding import BERTembedding

def load_bert_embedder(model_name="bert-base-uncased"):
    """
    Load the BERT embedding model from nlptechbook/BERTembeddings.
    """
    embedder = BERTembedding(model_name)
    return embedder

def get_bert_based_embeddings(texts, embedder):
    """
    Generate BERT-based embeddings for a list of texts using BERTembedding.
    """
    embeddings = embedder(texts)  # Directly get embeddings
    return embeddings
