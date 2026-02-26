# import torch
# import torch.nn as nn
# import torch.optim as optim
# import json
# from model import TinyLSTM
# from utils import preprocess_text, tokenize, build_vocab

# # Device (CPU / GPU safe)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Step 1: Preprocess dataset
# text = preprocess_text("dataset/stories.txt")

# # Step 2: Tokenize
# tokens = tokenize(text)

# # Step 3: Build vocabulary
# word2idx, idx2word = build_vocab(tokens)

# #  SAVE VOCAB (IMPORTANT)
# with open("vocab.json", "w") as f:
#     json.dump(word2idx, f)

# vocab_size = len(word2idx)

# # Step 4: Encode tokens
# encoded = [word2idx[w] for w in tokens]

# seq_length = 30
# inputs = []
# targets = []

# for i in range(len(encoded) - seq_length):
#     inputs.append(encoded[i:i+seq_length])
#     targets.append(encoded[i+1:i+seq_length+1])

# input_seq = torch.tensor(inputs).to(device)
# target_seq = torch.tensor(targets).to(device)

# # Step 5: Initialize model
# model = TinyLSTM(vocab_size).to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.003)

# # Step 6: Train
# for epoch in range(500):
#     optimizer.zero_grad()

#     output, _ = model(input_seq)

#     loss = criterion(
#         output.view(-1, vocab_size),
#         target_seq.view(-1)
#     )

#     loss.backward()
#     optimizer.step()

#     if epoch % 20 == 0:
#         print("Epoch:", epoch, "Loss:", loss.item())

# #  Save both model + vocab size
# torch.save({
#     "model_state_dict": model.state_dict(),
#     "vocab_size": vocab_size
# }, "tiny_lstm.pth")

# print("Training complete.")


import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader, TensorDataset

from model import TinyLSTM
from utils import preprocess_text, tokenize, build_vocab


# -----------------------
# Device Configuration
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------
# Step 1: Preprocess Dataset
# -----------------------
text = preprocess_text("dataset/stories.txt")


# -----------------------
# Step 2: Tokenize
# -----------------------
tokens = tokenize(text)


# -----------------------
# Step 3: Build Vocabulary
# -----------------------
word2idx, idx2word = build_vocab(tokens)

with open("vocab.json", "w") as f:
    json.dump(word2idx, f)

vocab_size = len(word2idx)
print("Vocabulary size:", vocab_size)


# -----------------------
# Step 4: Encode Tokens
# -----------------------
encoded = [word2idx[w] for w in tokens]

seq_length = 60  # Longer context for structured stories

inputs = []
targets = []

for i in range(len(encoded) - seq_length):
    inputs.append(encoded[i:i+seq_length])
    targets.append(encoded[i+1:i+seq_length+1])

input_seq = torch.tensor(inputs)
target_seq = torch.tensor(targets)


# -----------------------
# Step 5: DataLoader (Mini-batch Training)
# -----------------------
dataset = TensorDataset(input_seq, target_seq)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


# -----------------------
# Step 6: Initialize Model
# -----------------------
model = TinyLSTM(vocab_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)


# -----------------------
# Step 7: Training Loop
# -----------------------
epochs = 10

for epoch in range(epochs):

    total_loss = 0

    for batch_inputs, batch_targets in loader:

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()

        output, _ = model(batch_inputs)

        loss = criterion(
            output.view(-1, vocab_size),
            batch_targets.view(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 20 == 0:
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")


# -----------------------
# Step 8: Save Model
# -----------------------
torch.save({
    "model_state_dict": model.state_dict(),
    "vocab_size": vocab_size
}, "tiny_lstm.pth")

print("Training complete.")