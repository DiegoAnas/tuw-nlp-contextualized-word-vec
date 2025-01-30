import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split

# Constants
MAX_SEQ_LEN = 40
MAX_VOCAB_SIZE = 50000
EMBEDDING_DIM = 300
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'
DATA_DIR = "./data/imdb"
OUTPUT_DIR = "./data/processed"
TOKENIZER_PATH = os.path.join(OUTPUT_DIR, "imdb_tokenizer.pkl")

# Function to download and save raw IMDB dataset with fraction
def download_and_save_imdb(output_dir="./IMDB/data", fraction=1.0):
    dataset = load_dataset("imdb", split={"train": "train", "test": "test"})  # Removed unsupervised split

    def save_split(split, split_name):
        file_path = os.path.join(output_dir, f"{split_name}.txt")
        os.makedirs(output_dir, exist_ok=True)

        # Limit dataset size based on fraction
        subset_size = int(len(split) * fraction)
        split = split.select(range(subset_size))

        with open(file_path, "w", encoding="utf-8") as f:
            for example in split:
                f.write(f"{example['label']}\t{example['text']}\n")
        print(f"Saved {split_name} data to {file_path}")

    save_split(dataset["train"], "train")
    save_split(dataset["test"], "test")

# Function to read the raw IMDB dataset
def readdata(data_dir):
    paths = {
        "train": os.path.join(data_dir, "train.txt"),
        "test": os.path.join(data_dir, "test.txt"),
    }
    data = {}
    for key, path in paths.items():
        with open(path, "r", encoding="utf-8") as file:
            data[key] = file.readlines()
    return data["train"], data["test"]

# Helper functions for cleaning
def to_lowercase(text):
    return text.lower()

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def replace_numbers(text):
    return re.sub(r'\d+', 'n', text)

def truncate_text(text, max_length=200):
    return " ".join(text.split()[:max_length])

def remove_html_tags(text):
    text = re.sub(r'<br\s*/?>', ' ', text)  # Replace <br> with space
    text = re.sub(r'<.*?>', '', text)  # Remove other HTML tags
    return text

def remove_special_chars(text):
    text = re.sub(r'[^\w\s]', '', text)  # Removes punctuation
    return text

def remove_remaining_br_tags(text):
    text = re.sub(r'<br\s*/?>', ' ', text)  # Explicitly remove any remaining <br> tags
    return text

# Function to clean data and save it as cleaned .txt files
def clean_and_save(data, max_seq_len=200, output_dir="./IMDB/data/cleaned"):
    cleaned_data = []
    
    # Make output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for line in data:
        label, text = line.strip().split("\t", 1)

        # Clean the text
        text = remove_html_tags(text)
        text = remove_special_chars(text)
        text = to_lowercase(text)
        text = replace_numbers(text)
        text = remove_stopwords(text)
        text = truncate_text(text, max_seq_len)
        text = remove_remaining_br_tags(text)
        
        cleaned_data.append((int(label), text))
    
    # Save the cleaned data to a .txt file
    with open(os.path.join(output_dir, "cleaned_data.txt"), "w", encoding="utf-8") as f:
        for label, text in cleaned_data:
            f.write(f"{label}\t{text}\n")
    
    return cleaned_data

# Tokenization and saving the tokenized data
# Tokenization and saving the tokenized data
def tokenize_and_save(data, output_path, tokenizer, max_length=MAX_SEQ_LEN):
    texts = [text for label, text in data]
    labels = [label for label, text in data]
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, truncating='post', padding='post', value=0)

    with open(output_path, "w", encoding="utf-8") as f:
        for label, sequence in zip(labels, padded_sequences):
            f.write(f"{label}\t{' '.join(map(str, sequence))}\n")

    return padded_sequences, labels

# Main preprocessing and saving function
def preprocess_imdb(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, fraction=0.1):
    if not os.path.exists(data_dir):
        print(" Dataset directory not found. Downloading imdb...")
        download_and_save_imdb(data_dir, fraction)
    train, test = readdata(data_dir)

    # Clean data and save with fraction
    train_cleaned = clean_and_save(train, output_dir=os.path.join(output_dir, "train"))
    test_cleaned = clean_and_save(test, output_dir=os.path.join(output_dir, "test"))

    # Apply fraction concept: Reduce data size based on fraction
    train_cleaned = train_cleaned[:int(len(train_cleaned) * fraction)]
    test_cleaned = test_cleaned[:int(len(test_cleaned) * fraction)]

    # Split the training data into training and validation (80% train, 20% validation)
    train_cleaned, val_cleaned = train_test_split(train_cleaned, test_size=0.2, random_state=42)

    # Fit tokenizer
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, lower=True, oov_token=UNK_TOKEN)
    tokenizer.fit_on_texts([text for label, text in train_cleaned + val_cleaned + test_cleaned])

    # Save tokenized data
    os.makedirs(output_dir, exist_ok=True)
    train_seq, train_labels = tokenize_and_save(train_cleaned, os.path.join(output_dir, "train.tok.txt"), tokenizer)
    val_seq, val_labels = tokenize_and_save(val_cleaned, os.path.join(output_dir, "val.tok.txt"), tokenizer)
    test_seq, test_labels = tokenize_and_save(test_cleaned, os.path.join(output_dir, "test.tok.txt"), tokenizer)

    # Save the validation data
    with open(os.path.join(output_dir, "val.tok.txt"), "w", encoding="utf-8") as f:
        for label, sequence in zip(val_labels, val_seq):
            f.write(f"{label}\t{' '.join(map(str, sequence))}\n")

    return train_seq, train_labels, val_seq, val_labels, test_seq, test_labels, tokenizer.word_index


if __name__ == "__main__":
    preprocess_imdb()
