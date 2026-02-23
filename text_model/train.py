import torch
import torch.nn as nn
import torch.optim as optim
import json
from model import TinyLSTM
from utils import preprocess_text, tokenize, build_vocab

# Device (CPU / GPU safe)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Preprocess dataset
text = preprocess_text("dataset/stories.txt")

# Step 2: Tokenize
tokens = tokenize(text)

# Step 3: Build vocabulary
word2idx, idx2word = build_vocab(tokens)

# 🔥 SAVE VOCAB (IMPORTANT)
with open("vocab.json", "w") as f:
    json.dump(word2idx, f)

vocab_size = len(word2idx)

# Step 4: Encode tokens
encoded = [word2idx[w] for w in tokens]

seq_length = 10
inputs = []
targets = []

for i in range(len(encoded) - seq_length):
    inputs.append(encoded[i:i+seq_length])
    targets.append(encoded[i+1:i+seq_length+1])

input_seq = torch.tensor(inputs).to(device)
target_seq = torch.tensor(targets).to(device)

# Step 5: Initialize model
model = TinyLSTM(vocab_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Step 6: Train
for epoch in range(300):
    optimizer.zero_grad()

    output, _ = model(input_seq)

    loss = criterion(
        output.view(-1, vocab_size),
        target_seq.view(-1)
    )

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print("Epoch:", epoch, "Loss:", loss.item())

# 🔥 Save both model + vocab size
torch.save({
    "model_state_dict": model.state_dict(),
    "vocab_size": vocab_size
}, "tiny_lstm.pth")

print("Training complete.")
