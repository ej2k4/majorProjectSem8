import json
import os
import re
import torch
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

# precompiled tokenizer regex (faster)
TOKEN_PATTERN = re.compile(r"\b\w+\b")


# -----------------------
# Vocabulary
# -----------------------

class Vocabulary:

    def __init__(self):

        # special tokens
        self.special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

        self.word2idx = {}
        self.idx2word = {}
        self.size = 0

    def build(self, texts):

        words = set()

        for text in texts:
            tokens = TOKEN_PATTERN.findall(text.lower())
            words.update(tokens)

        words = sorted(list(words))

        # add special tokens first
        all_tokens = self.special_tokens + words

        self.word2idx = {w: i for i, w in enumerate(all_tokens)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        self.size = len(all_tokens)


# -----------------------
# Dataset loader
# -----------------------

def load_dataset():

    texts = []

    files = [
        "data/stories.json",
        "outputs/generated_dataset.json"
    ]

    for path in files:

        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:

            try:
                data = json.load(f)
            except:
                continue

            # JSON list
            if isinstance(data, list):

                for item in data:

                    if isinstance(item, dict):

                        for value in item.values():
                            if isinstance(value, str) and value.strip():
                                texts.append(value)

                    elif isinstance(item, str):
                        texts.append(item)

            # JSON dictionary
            elif isinstance(data, dict):

                for value in data.values():

                    if isinstance(value, list):

                        for v in value:
                            if isinstance(v, str):
                                texts.append(v)

                    elif isinstance(value, str):
                        texts.append(value)

    if len(texts) == 0:
        raise ValueError("Dataset empty. Check JSON files.")

    return texts


# -----------------------
# Build vocab
# -----------------------

def build_vocab(texts):

    vocab = Vocabulary()
    vocab.build(texts)

    return vocab


# -----------------------
# Save vocab
# -----------------------

def save_vocab(vocab, path="vocab.json"):

    data = {
        "word2idx": vocab.word2idx,
        "idx2word": vocab.idx2word,
        "size": vocab.size
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# -----------------------
# Create sequences
# -----------------------

def create_sequences(texts, vocab, seq_len):

    sequences = []

    stride = 5

    bos = vocab.word2idx["<BOS>"]
    eos = vocab.word2idx["<EOS>"]

    for text in texts:

        tokens = TOKEN_PATTERN.findall(text.lower())

        token_ids = [vocab.word2idx[t] for t in tokens if t in vocab.word2idx]

        # add BOS/EOS
        token_ids = [bos] + token_ids + [eos]

        for i in range(0, len(token_ids) - seq_len, stride):

            x = token_ids[i:i + seq_len]

            sequences.append(x)

    return sequences


# -----------------------
# DataLoader
# -----------------------

def get_dataloader(sequences, batch_size=16):

    if len(sequences) == 0:
        raise ValueError("No training sequences found. Check dataset format.")

    x = torch.tensor(sequences, dtype=torch.long)

    dataset = TensorDataset(x)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    return loader


# -----------------------
# Save model
# -----------------------

def save_model(model, path="model.pt"):

    torch.save(model.state_dict(), path)


# -----------------------
# JSON training log
# -----------------------

def save_json_log(data, file_path="training_log.json"):

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "data": data
    }

    try:
        with open(file_path, "r") as f:
            existing = json.load(f)
    except:
        existing = []

    existing.append(record)

    with open(file_path, "w") as f:
        json.dump(existing, f, indent=4)

