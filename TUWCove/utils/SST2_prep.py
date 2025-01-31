import os
import numpy as np
import re
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
DATA_DIR = "./data/sst2"
OUTPUT_DIR = "./data/processed"
TOKENIZER_PATH = os.path.join(OUTPUT_DIR, "wmt16_tokenizer.pkl")

def download_and_save_sst2(output_dir="./SST2/data", fraction=1.0):
    """
    Downloads SST-2 dataset and saves it to output directory.
    
    Args:
        output_dir (str): Directory to save the data.
        fraction (float): Fraction of the dataset to save. Default is 1.0 (save all).
    """
    dataset = load_dataset("glue", "sst2", split={"train": "train", "validation": "validation", "test": "test"})
    os.makedirs(output_dir, exist_ok=True)

    def save_split(split, split_name):
        """
        Saves a dataset split while limiting size based on `fraction`.
        
        Args:
            split (Dataset): The dataset split to save (train, val, test).
            split_name (str): The name of the split (train, val, test).
        """
        # Limit dataset size based on fraction
        subset_size = int(len(split) * fraction)
        split = split.select(range(subset_size))
        
        file_path = os.path.join(output_dir, f"{split_name}.tsv")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("sentence\tlabel\n")
            for example in split:
                f.write(f"{example['sentence']}\t{example['label']}\n")
        print(f"Saved {split_name} data to {file_path}")

        # Print a sample of the data
        print(f"Sample from {split_name}: {split[0]}")  # Prints the first example

    save_split(dataset["train"], "train")
    save_split(dataset["validation"], "val")
    save_split(dataset["test"], "test")


def readdata(data_dir):
    """
    Reads SST-2 dataset from the specified directory.

    Args:
        data_dir (str): Path to the dataset directory.

    Returns:
        tuple: Training, validation, and test data for sentences and labels.
    """
    paths = {
        "train": os.path.join(data_dir, "train.tsv"),
        "test": os.path.join(data_dir, "test.tsv"),
        "val": os.path.join(data_dir, "val.tsv"),
    }

    data = {}
    for key, path in paths.items():
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()[1:]  # Skip header
            sentences = []
            labels = []
            for line in lines:
                parts = line.strip().split("\t")
                sentences.append(parts[0])
                labels.append(int(parts[1]))
            data[key] = (sentences, labels)
    return data["train"][0], data["train"][1], data["test"][0], data["test"][1], data["val"][0], data["val"][1]


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

def preprocess_sst2(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, fraction=0.1):
    """
    Main function to preprocess SST-2 data.

    Args:
        data_dir (str): Path to SST-2 dataset directory.
        output_dir (str): Path to save processed data.

    Returns:
        tuple: Processed data and vocabularies.
    """

    if not os.path.exists(data_dir):
        print(" Dataset directory not found. Downloading sst2...")
        download_and_save_sst2(data_dir, fraction)

    train_sentences, train_labels, test_sentences, test_labels, val_sentences, val_labels = readdata(data_dir)

    # Clean data
    train_sentences = clean(train_sentences)
    val_sentences = clean(val_sentences)
    test_sentences = clean(test_sentences)

    # Fit tokenizer
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, lower=True, oov_token=UNK_TOKEN)
    tokenizer.fit_on_texts(train_sentences + val_sentences + test_sentences)

    # Save tokenized data
    os.makedirs(output_dir, exist_ok=True)
    train_seq = tokenize_and_save(train_sentences, os.path.join(output_dir, "train.tok"), tokenizer)
    val_seq = tokenize_and_save(val_sentences, os.path.join(output_dir, "val.tok"), tokenizer)
    test_seq = tokenize_and_save(test_sentences, os.path.join(output_dir, "test.tok"), tokenizer)

    return train_seq, train_labels, val_seq, val_labels, test_seq, test_labels, tokenizer.word_index


if __name__ == "__main__":
    preprocess_sst2()
