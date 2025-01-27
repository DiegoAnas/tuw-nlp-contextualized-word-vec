import torch
import numpy as np
from transformers import AutoTokenizer  # For GloVe tokenization
from cove import CoVeEncoder  # Pre-trained CoVe encoder (use your implementation)

def load_glove_embeddings(glove_path, vocab, embedding_dim=300):
    """
    Load GloVe embeddings and create an embedding matrix.
    
    Args:
        glove_path (str): Path to the GloVe file.
        vocab (dict): Vocabulary mapping (word -> index).
        embedding_dim (int): Dimension of GloVe embeddings (default=300).
        
    Returns:
        torch.Tensor: A tensor of GloVe embeddings.
    """
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    # Create an embedding matrix
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, idx in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
        else:
            embedding_matrix[idx] = np.random.uniform(-0.1, 0.1, embedding_dim)  # Random for unknown words
    
    return torch.FloatTensor(embedding_matrix)

def prepare_combined_embeddings(vocab, glove_path, cove_encoder, embedding_dim=300):

    glove_embeddings = load_glove_embeddings(glove_path, vocab, embedding_dim)
    
    # Generate CoVe embeddings for the vocabulary
    # Prepare input for CoVe (vocabulary tokens)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    vocab_tokens = [tokenizer.decode([idx]) for idx in range(len(vocab))]  # Convert vocab indices to tokens
    
    # Tokenize and convert tokens into IDs
    inputs = tokenizer(vocab_tokens, padding=True, truncation=True, max_length=32, return_tensors="pt")
    
    # Get CoVe embeddings using the pre-trained CoVe encoder
    with torch.no_grad():
        cove_embeddings = cove_encoder(inputs["input_ids"])  # CoVeEncoder forward pass
    
    # Step 3: Concatenate GloVe and CoVe embeddings
    combined_dim = embedding_dim + cove_embeddings.size(-1)
    combined_embeddings = torch.cat((glove_embeddings, cove_embeddings), dim=1)
    
    return combined_embeddings



