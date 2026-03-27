import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from model import Encoder, Decoder, Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Load Dataset
# -------------------
df = pd.read_csv(r"asd_dataset.csv")

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

dataset = ASDDataset(df)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# -------------------
# Initialize Model
# -------------------
vocab_size = len(word2idx)

encoder = Encoder(vocab_size, 128, 256)
decoder = Decoder(vocab_size, 128, 256)
model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<pad>"])

# -------------------
# Training Loop
# -------------------
for epoch in range(10):
    model.train()
    total_loss = 0

    for src, trg in loader:
        src, trg = src.to(device), trg.to(device)

        output = model(src, trg)

        output = output[:, 1:].reshape(-1, vocab_size)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader)}")

torch.save(model.state_dict(), "asd_model.pt")
print("Training complete.")
import pickle

with open("vocab.pkl", "wb") as f:
    pickle.dump(word2idx, f)
    