import torch
from model import TinyLSTM

vocab_size = 100

model = TinyLSTM(vocab_size)

x = torch.randint(0, vocab_size, (2, 5))  # batch_size=2, seq_len=5

output, hidden = model(x)

print("Output shape:", output.shape)
