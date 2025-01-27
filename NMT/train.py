import os
import torch
import torch.nn as nn
import torch.optim as optim
from models import Encoder, Decoder, NMTModel
from dataloader import get_dataloaders
from Data_prep_WMT16 import preprocess_wmt

def train_model(train_loader, val_loader, model, optimizer, loss_fn, num_epochs, device):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for src_batch, tgt_batch in train_loader:
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model((src_batch, tgt_batch[:, :-1]))  # Exclude last token for inputs
            loss = loss_fn(output.view(-1, output.size(-1)), tgt_batch[:, 1:].contiguous().view(-1))  # Ignore <BOS>

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for src_batch, tgt_batch in val_loader:
                src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
                output = model((src_batch, tgt_batch[:, :-1]))
                loss = loss_fn(output.view(-1, output.size(-1)), tgt_batch[:, 1:].contiguous().view(-1))
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")

if __name__ == "__main__":
    # Paths
    data_dir = "./NMT/data/processed"
    glove_path = "./NMT/data/glove.840B.300d.txt"

    # Preprocess data to get vocab and embedding matrix
    _, _, _, _, _, _, embedding_matrix, vocab, _ = preprocess_wmt(data_dir, glove_path, data_dir)

    # Hyperparameters
    batch_size = 64
    num_epochs = 10
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    encoder = Encoder(
        num_layers=2, 
        bidirectional=True, 
        dropout=0.2, 
        rnn_size=512, 
        word_vec_dim=300, 
        dict_size=50000, 
        glove_path=glove_path, 
        vocab=vocab
    )
    decoder = Decoder(num_layers=2, bidirectional=False, dropout=0.2, rnn_size=512, word_vec_dim=300, dict_size=50000)
    model = NMTModel(encoder=encoder, decoder=decoder, rnn_size=512, tgt_dict_size=50000)

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens

    # DataLoader
    train_src = os.path.join(data_dir, "train.tok.en")
    train_tgt = os.path.join(data_dir, "train.tok.de")
    val_src = os.path.join(data_dir, "val.tok.en")
    val_tgt = os.path.join(data_dir, "val.tok.de")

    train_loader, val_loader = get_dataloaders(train_src, train_tgt, val_src, val_tgt, batch_size)

    # Train
    train_model(train_loader, val_loader, model, optimizer, loss_fn, num_epochs, device)
