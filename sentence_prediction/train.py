import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from model import Encoder, Decoder, Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Load Dataset
# -------------------
df = pd.read_csv("asd_train.csv")

# VALIDATION SPLIT
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

sentences = list(df["fragmented_input"]) + list(df["corrected_output"])

# -------------------
# Build Vocabulary
# -------------------
words = set()
for sentence in sentences:
    for word in sentence.lower().split():
        words.add(word)

word2idx = {word: idx+4 for idx, word in enumerate(words)}
word2idx["<pad>"] = 0
word2idx["<sos>"] = 1
word2idx["<eos>"] = 2
word2idx["<unk>"] = 3

idx2word = {i: w for w, i in word2idx.items()}

# -------------------
# Convert text to numbers
# -------------------
def numericalize(sentence):
    return [word2idx.get(word, word2idx["<unk>"]) for word in sentence.lower().split()]

# -------------------
# Dataset Class
# -------------------
class ASDDataset(Dataset):
    def __init__(self, df, max_len=15):
        self.df = df
        self.max_len = max_len

    def pad(self, seq):
        seq = seq[:self.max_len]
        seq += [word2idx["<pad>"]] * (self.max_len - len(seq))
        return seq

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        inp = numericalize(self.df.iloc[idx]["fragmented_input"])
        out = [word2idx["<sos>"]] + numericalize(self.df.iloc[idx]["corrected_output"]) + [word2idx["<eos>"]]

        inp = self.pad(inp)
        out = self.pad(out)

        return torch.tensor(inp), torch.tensor(out)

# TRAIN + VALIDATION DATASETS
train_dataset = ASDDataset(train_df)
val_dataset = ASDDataset(val_df)

# LOADERS
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# -------------------
# Initialize Model
# -------------------
vocab_size = len(word2idx)

encoder = Encoder(vocab_size, 128, 256)
decoder = Decoder(vocab_size, 128, 256)
model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<pad>"])

# LOSS TRACKING
train_losses = []
val_losses = []

# -------------------
# EARLY STOPPING SETUP (FIXED)
# -------------------
best_val_loss = float('inf')
patience = 2
wait = 0

# -------------------
# Training Loop
# -------------------
for epoch in range(15):
    model.train()
    total_train_loss = 0

    for src, trg in train_loader:
        src, trg = src.to(device), trg.to(device)

        output = model(src, trg)

        output = output[:, 1:].reshape(-1, vocab_size)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # VALIDATION
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for src, trg in val_loader:
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg)

            output = output[:, 1:].reshape(-1, vocab_size)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # -------------------
    # EARLY STOPPING (FIXED PROPERLY)
    # -------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pt")
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered")
            break

# -------------------
# LOAD BEST MODEL (IMPORTANT)
# -------------------
print("Loading best model...")
model.load_state_dict(torch.load("best_model.pt"))

# SAVE FINAL MODEL
torch.save(model.state_dict(), "asd_model.pt")
print("Training complete.")

# SAVE VOCAB
import pickle
with open("vocab.pkl", "wb") as f:
    pickle.dump(word2idx, f)