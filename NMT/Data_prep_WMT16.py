import os
import numpy as np
import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset

# Constants
MAX_SEQ_LEN = 40
MAX_VOCAB_SIZE = 50000
EMBEDDING_DIM = 300
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'


def download_and_save_wmt16(output_dir: str="./NMT/data/wmt16", shards:int=1):
    """_summary_

    Args:
        output_dir (str, optional): _description_. Defaults to "./NMT/data/wmt16".
        fraction (int, optional): Dataset gets divided into these many shards
                                    e.g for a 1% pass shards=100. Defaults to 1.
    """
    #Downloads WMT16 English-German dataset and saves it to output directory.

    dataset = load_dataset("wmt16", "de-en", split={"train": "train", "validation": "validation", "test": "test"})
        
    def save_split(split, lang, split_name):
        file_path = os.path.join(output_dir, f"{split_name}.{lang}")
        os.makedirs(output_dir, exist_ok=True)
        # Limit dataset size
        if shards != 1:
            fraction = 1 / shards
            subset_size = int(len(split) * fraction)
            split = split.select(range(subset_size))
        with open(file_path, "w", encoding="utf-8") as f:
            for example in split:
                if lang not in example["translation"]:
                    print(f"Missing key '{lang}' in example: {example}")
                    continue
                f.write(example["translation"][lang] + "\n")
        print(f"Saved {split_name} data for '{lang}' to {file_path}")

    # Save the train, validation, and test splits
    save_split(dataset["train"], "en", "train")
    save_split(dataset["train"], "de", "train")
    save_split(dataset["validation"], "en", "val")
    save_split(dataset["validation"], "de", "val")
    save_split(dataset["test"], "en", "test")
    save_split(dataset["test"], "de", "test")


def readdata(data_dir):
    """
    Reads WMT16 dataset from the specified directory.

    Args:
        data_dir (str): Path to the dataset directory.

    Returns:
        tuple: Training, validation, and test data for English and German.
    """
    paths = {
        "train_en": os.path.join(data_dir, "train.en"),
        "train_de": os.path.join(data_dir, "train.de"),
        "test_en": os.path.join(data_dir, "test.en"),
        "test_de": os.path.join(data_dir, "test.de"),
        "val_en": os.path.join(data_dir, "val.en"),
        "val_de": os.path.join(data_dir, "val.de"),
    }

    data = {}
    for key, path in paths.items():
        with open(path, "r", encoding="utf-8") as file:
            data[key] = file.readlines()
    return data["train_en"], data["train_de"], data["test_en"], data["test_de"], data["val_en"], data["val_de"]

def save_tokenizer(tokenizer, filepath):
    """Saves the tokenizer to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f" Tokenizer saved to {filepath}")


def load_tokenizer(filepath):
    """Loads the tokenizer from disk."""
    with open(filepath, "rb") as f:
        tokenizer = pickle.load(f)
    print(f" Tokenizer loaded from {filepath}")
    return tokenizer

def clean(data, max_seq_len=MAX_SEQ_LEN):
    """
    Cleans the input data by applying preprocessing steps such as token replacements.

    Args:
        data (list): Input text data.
        max_seq_len (int): Maximum sequence length.

    Returns:
        list: Cleaned sentences with <BOS> and <EOS> tokens.
    """
    cleaned_data = []
    for line in data:
        line = re.sub(r'[0-9]+p*', 'n', line)  # Replace all numbers with 'n'
        line = re.sub(r'\s+', ' ', line)  # Remove extra spaces
        line = re.sub("'", '', line)  # Remove apostrophes
        tokens = line.strip().split()
        cleaned_data.append(f"{BOS_TOKEN} {' '.join(tokens[:max_seq_len])} {EOS_TOKEN}")
    return cleaned_data


def tokenize_and_save(data, output_path, tokenizer, max_length=MAX_SEQ_LEN):
    """
    Tokenizes and pads data, then saves it to a file.

    Args:
        data (list): List of sentences.
        output_path (str): Path to save the tokenized data.
        tokenizer (Tokenizer): Fitted tokenizer.
        max_length (int): Maximum sequence length.

    Returns:
        np.ndarray: Tokenized and padded data.
    """
    sequences = tokenizer.texts_to_sequences(data)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, truncating='post', padding='post', value=0)

    with open(output_path, "w", encoding="utf-8") as f:
        for sequence in padded_sequences:
            f.write(" ".join(map(str, sequence)) + "\n")

    return padded_sequences


def create_embedding_indexmatrix(glove_path, vocab, max_words, embedding_dim=300) -> np.ndarray:
    """
    Creates an embedding matrix using GloVe embeddings.

    Args:
        glove_path (str): Path to GloVe embeddings.
        vocab (dict): Vocabulary from the tokenizer.
        max_words (int): Maximum number of words to include in the matrix.
        embedding_dim (int): Dimension of the embeddings.

    Returns:
        np.ndarray: Embedding matrix.
    """
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((max_words, embedding_dim))
    embedding_matrix[0] = np.random.uniform(-0.05, 0.05, embedding_dim)  # For <unk>
    for word, idx in vocab.items():
        if idx < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector
            else:
                embedding_matrix[idx] = np.random.uniform(-0.05, 0.05, embedding_dim)  # Random for OOV words

    return embedding_matrix


def preprocess_wmt(data_dir, glove_path, output_dir):
    """
    Main function to preprocess WMT16 data and prepare embeddings.

    Args:
        data_dir (str): Path to WMT dataset directory.
        glove_path (str): Path to GloVe embeddings.
        output_dir (str): Path to save processed data.

    Returns:
        tuple: Processed data, embedding matrix, and vocabularies.
    """
    train_en, train_de, test_en, test_de, val_en, val_de = readdata(data_dir)

    # Clean data
    train_en = clean(train_en)
    train_de = clean(train_de)
    test_en = clean(test_en)
    test_de = clean(test_de)
    val_en = clean(val_en)
    val_de = clean(val_de)

    # Fit tokenizer
    tokenizer_en = Tokenizer(num_words=MAX_VOCAB_SIZE, lower=True, oov_token=UNK_TOKEN)
    tokenizer_en.fit_on_texts(train_en + val_en +  test_en )
    
    # Fit tokenizer
    tokenizer_de = Tokenizer(num_words=MAX_VOCAB_SIZE, lower=True, oov_token=UNK_TOKEN)
    tokenizer_de.fit_on_texts(train_de + val_de + test_de)

    # Save tokenized data
    os.makedirs(output_dir, exist_ok=True)
    train_en_seq = tokenize_and_save(train_en, os.path.join(output_dir, "train.tok.en"), tokenizer_en)
    train_de_seq = tokenize_and_save(train_de, os.path.join(output_dir, "train.tok.de"), tokenizer_de)
    val_en_seq = tokenize_and_save(val_en, os.path.join(output_dir, "val.tok.en"), tokenizer_en)
    val_de_seq = tokenize_and_save(val_de, os.path.join(output_dir, "val.tok.de"), tokenizer_de)
    test_en_seq = tokenize_and_save(test_en, os.path.join(output_dir, "test.tok.en"), tokenizer_en)
    test_de_seq = tokenize_and_save(test_de, os.path.join(output_dir, "test.tok.de"), tokenizer_de)

    # Prepare embeddings
    embedding_matrix = create_embedding_indexmatrix(
        glove_path=glove_path,
        vocab=tokenizer_en.word_index,
        max_words=MAX_VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM
    )
    
    with open(f"{output_dir}glove.embs.pickle", "wb") as f:
        pickle.dump(embedding_matrix, f)
    print(f" Gloves saved to {output_dir}glove.embs.pickle")
    
    save_tokenizer(tokenizer=tokenizer_en, filepath=f"{output_dir}tokenizer.en.pickle")
    save_tokenizer(tokenizer=tokenizer_de, filepath=f"{output_dir}tokenizer.de.pickle")

    return (
        train_en_seq, train_de_seq, val_en_seq, val_de_seq, test_en_seq, test_de_seq,
        embedding_matrix, tokenizer_en.word_index, {v: k for k, v in tokenizer_en.word_index.items()},
        tokenizer_en, tokenizer_de
    )

def preprocess(download_dir="./NMT/data/wmt16",
               data_dir="./NMT/data/wmt16",
               glove_path="./NMT/data/glove.840B.300d.txt",
               preprocessed_dir="./NMT/data/prep",
               shards=1):
    
    download_and_save_wmt16(output_dir=download_dir, shards=shards)
    train_en, train_de, val_en, val_de, test_en, test_de,\
        embedding_matrix,vocab, i2w, tokenizer_en, tokenizer_de \
        = preprocess_wmt(
            data_dir, glove_path, preprocessed_dir
            )

    # Print summary
    print(f"Preprocessing complete.")
    print(f"Training data (EN): {len(train_en)} sentences")
    print(f"Training data (DE): {len(train_de)} sentences")
    print(f"Validation data (EN): {len(val_en)} sentences")
    print(f"Validation data (DE): {len(val_de)} sentences")
    print(f"Test data (EN): {len(test_en)} sentences")
    print(f"Test data (DE): {len(test_de)} sentences")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"GloVe embedding matrix shape: {embedding_matrix.shape}")
    
    return train_en, train_de, val_en, val_de, test_en, test_de, embedding_matrix, vocab, i2w, tokenizer_en, tokenizer_de
    

if __name__ == "__main__":
    preprocess()