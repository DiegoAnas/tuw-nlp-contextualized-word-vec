import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import os

import TUWCove
from TUWCove.models.NMTLSTM import Encoder, Decoder, NMTModel
from TUWCove.utils.Data_prep_WMT16 import preprocess
import TUWCove.utils.constants as constants

# Paths
GLOVE_PATH = "./data/glove.6B.300d.txt"
VOCAB_PATH = "./data/prep/vocab.json"
EMBEDDINGS_PATH = "./data/prep/combined_embeddings.pth"
TRAIN_SRC_PATH = "./data/prep/train.tok.en"
TRAIN_TGT_PATH = "./data/prep/train.tok.de"
VAL_SRC_PATH = "./data/prep/val.tok.en"
VAL_TGT_PATH = "./data/prep/val.tok.de"
TEST_SRC_PATH = "./data/prep/test.tok.en"
TEST_TGT_PATH = "./data/prep/test.tok.de"
MODEL_SAVE_PATH = "./checkpoints/"

# Hyperparameters
EMBEDDING_DIM = 300
RNN_SIZE = 300
DROPOUT = 0.2
NUM_LAYERS = 2
BIDIRECTIONAL = True
PAD_IDX = constants.PAD
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Function to save checkpoints
def save_checkpoint(save_path, model, optimizer, epoch, loss, scheduler=None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    if scheduler:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at {save_path}")


# -------------------
# Load Vocabulary
# -------------------
def load_vocab(vocab_path):
    if not os.path.isfile(vocab_path):
        print(f"Vocab json file not found. Missing: {vocab_path}")
        return None
    """Loads the vocabulary from a JSON file."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Loaded vocabulary with {len(vocab)} words.")
    return vocab

# -----------------------
# Load Preprocessed Data
# -----------------------
def load_tokenized_data(src_path, tgt_path):
    """Loads tokenized sentences from file and ensures matching lengths."""
    if not os.path.isfile(src_path):
        print(f"Source sentences file not found. Missing: {src_path}")
        return None
    if not os.path.isfile(tgt_path):
        print(f"Target sentences file not found. Missing: {tgt_path}")
        return None
    with open(src_path, "r", encoding="utf-8") as f_src, open(tgt_path, "r", encoding="utf-8") as f_tgt:
        src_lines = f_src.readlines()
        tgt_lines = f_tgt.readlines()

    # Fix mismatched sentence count
    min_len = min(len(src_lines), len(tgt_lines))
    
    if len(src_lines) != len(tgt_lines):
        print(f"⚠ Warning: Mismatch in sentence count! {len(src_lines)} in source, {len(tgt_lines)} in target.")
        print(f"✂ Truncating to {min_len} to match both.")

    # Keep only matching sentences
    src_lines, tgt_lines = src_lines[:min_len], tgt_lines[:min_len]

    # Convert tokenized text to integer lists
    src_sentences = [[int(token) for token in line.strip().split()] for line in src_lines]
    tgt_sentences = [[int(token) for token in line.strip().split()] for line in tgt_lines]

    return src_sentences, tgt_sentences

# -----------------------
# Create DataLoader
# -----------------------
def create_dataloader(src_sentences, tgt_sentences, batch_size, pad_idx=PAD_IDX):
    """Creates a PyTorch DataLoader with padded sequences."""
    assert len(src_sentences) == len(tgt_sentences), "Mismatch between source and target sentences."

    # Convert lists to PyTorch tensors (padded sequences)
    src_tensors = [torch.tensor(sentence, dtype=torch.long) for sentence in src_sentences]
    tgt_tensors = [torch.tensor(sentence, dtype=torch.long) for sentence in tgt_sentences]

    # Pad sequences to the same length
    src_tensors = pad_sequence(src_tensors, batch_first=True, padding_value=pad_idx)
    tgt_tensors = pad_sequence(tgt_tensors, batch_first=True, padding_value=pad_idx)

    dataset = TensorDataset(src_tensors, tgt_tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ----------------------------
# Initialize Model
# ----------------------------
def initialize_model(src_vocab, tgt_vocab, combined_embeddings):
    """Initialize the NMT model with precomputed embeddings."""
    embedding_dim = combined_embeddings.size(1)

    encoder = Encoder(
        num_layers=NUM_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT,
        rnn_size=RNN_SIZE,
        word_vec_dim=embedding_dim,
        dict_size=len(src_vocab),
        vocab=src_vocab,
        glove_embeddings=combined_embeddings,  
        padding=PAD_IDX,
        freeze_embeddings=True
    )
    #encoder.embedding.weight.data.copy_(combined_embeddings)

    decoder = Decoder(
        num_layers=NUM_LAYERS,
        bidirectional=False,
        dropout=DROPOUT,
        rnn_size=RNN_SIZE,
        word_vec_dim=EMBEDDING_DIM, 
        dict_size=len(tgt_vocab),
        padding=PAD_IDX
    )

    model = NMTModel(
        encoder=encoder,
        decoder=decoder,
        rnn_size=RNN_SIZE,
        tgt_dict_size=len(tgt_vocab),
        dropout=DROPOUT
    )

    return model

# -----------------------
# Training Loop
# -----------------------
def train_model(model, train_dl, validation_dl, tgt_vocab_size, epochs=EPOCHS, lr=LEARNING_RATE):
    """Trains the model using mini-batch gradient descent."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)  # Ignore padding in loss
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for src_batch, tgt_batch in train_dl:
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)

            # Forward pass
            output = model((src_batch, tgt_batch))  # Shape: [batch_size, seq_len, vocab_size]
            
            # Ensure output and target are flattened for CrossEntropyLoss
            output = output.view(-1, tgt_vocab_size)  
            tgt_batch = tgt_batch.view(-1)  

            # Compute loss while ignoring padding
            loss = criterion(output, tgt_batch)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for src_batch, tgt_batch in validation_dl:
                src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
                output = model((src_batch, tgt_batch[:, :-1]))
                loss = criterion(output.view(-1, output.size(-1)), tgt_batch[:, 1:].contiguous().view(-1))
                val_loss += loss.item()
            avg_val_loss = val_loss / len(validation_dl)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_checkpoint_path = os.path.join(MODEL_SAVE_PATH, "best_checkpoint.pth")
                save_checkpoint(best_checkpoint_path, model, optimizer, epoch + 1, best_val_loss, scheduler=scheduler)
                print(f"Best model updated with Validation Loss: {best_val_loss:.4f}")
                
    # Save model
    final_checkpoint_path = os.path.join(MODEL_SAVE_PATH, "final_checkpoint.pth")
    save_checkpoint(final_checkpoint_path, model, optimizer, epoch + 1, best_val_loss, scheduler=scheduler)
    print(f" Model saved to {final_checkpoint_path}")

# -----------------------
# Run Full Training
# -----------------------
def main():
    parser = argparse.ArgumentParser(description='Train base MTLSTM network')
    parser.add_argument('-download_dir', default="./NMT/data/wmt16", type=str, help='directory to download data to')
    parser.add_argument('-data_dir', default="./NMT/data/wmt16", type=str, help='directory to store dataset')    
    parser.add_argument('-glove_path', default="./NMT/data/glove.840B.300d.txt", type=str, help='GLoVE embeddings file')
    parser.add_argument('-preprocessed_dir', default="./NMT/data/processed", type=str, help='directory where preprocessed data is stored')    
    parser.add_argument('-shards', default=100, type=int, help='specify shards of dataset to use')    
    """# Paths TODO add to argparse
    GLOVE_PATH = "./data/glove.840B.300d.txt"
    VOCAB_PATH = "./data/processed/vocab.json"
    EMBEDDINGS_PATH = "./data/processed/combined_embeddings.pth"
    TRAIN_SRC_PATH = "./data/processed/train.tok.en"
    TRAIN_TGT_PATH = "./data/processed/train.tok.de"
    MODEL_SAVE_PATH = "./data/processed/nmt_model.pth"
    """
    args = parser.parse_args()
    
    if args.preprocess:
        preprocess(download_dir=args.download_dir,
                   data_dir=args.data_dir,
                   glove_path=args.glove_path,
                   preprocessed_dir=args.preprocessed_dir, 
                   shards=args.shards)
    
    print("\n Loading vocabulary...")
    vocab = load_vocab(args.glove_path)
    
    print("\n Loading precomputed embeddings...")
    combined_embeddings = torch.load(EMBEDDINGS_PATH)

    print("\n Loading tokenized dataset...")
    src_sentences, tgt_sentences = load_tokenized_data(TRAIN_SRC_PATH, TRAIN_TGT_PATH)

    print("\n Creating DataLoader...")
    dataloader = create_dataloader(src_sentences, tgt_sentences, BATCH_SIZE)
    
    val_src_sentences, val_tgt_sentences = load_tokenized_data(VAL_SRC_PATH, VAL_TGT_PATH)
    val_dl = create_dataloader(val_src_sentences, val_tgt_sentences, BATCH_SIZE)

    print("\n Initializing model...")
    model = initialize_model(vocab, combined_embeddings)

    print("\n Starting training...")
    train_model(model, dataloader, validation_dl= val_dl, vocab_size=len(vocab))

if __name__ == "__main__":
    main()