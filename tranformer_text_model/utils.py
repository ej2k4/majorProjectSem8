import json
import torch
from torch.utils.data import Dataset, DataLoader

SEQ_LEN = 64
STRIDE = 4


def load_dataset(path="data/stories.json"):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def build_vocab(texts):

    tokens = set()

    for t in texts:
        tokens.update(t.split())

    tokens = sorted(list(tokens))

    word2idx = {w: i for i, w in enumerate(tokens)}
    idx2word = {i: w for w, i in word2idx.items()}

    vocab = {
        "word2idx": word2idx,
        "idx2word": idx2word,
        "size": len(word2idx)
    }

    with open("vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)

    return word2idx, idx2word


def encode(text, word2idx):

    return [word2idx[w] for w in text.split() if w in word2idx]


class StoryDataset(Dataset):

    def __init__(self, sequences):
        self.data = sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        seq = self.data[idx]

        x = torch.tensor(seq[:-1])
        y = torch.tensor(seq[1:])

        return x, y


def build_sequences(texts, word2idx):

    sequences = []

    for t in texts:

        enc = encode(t, word2idx)

        if len(enc) < SEQ_LEN:
            continue

        for i in range(0, len(enc) - SEQ_LEN, STRIDE):

            sequences.append(enc[i:i + SEQ_LEN + 1])

    print("Total training sequences:", len(sequences))

    return sequences


def get_dataloader(seqs, batch_size=16):

    dataset = StoryDataset(seqs)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return loader
