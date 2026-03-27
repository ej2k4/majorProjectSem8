# import torch
# import torch.nn as nn
# import torch.optim as optim
# import os
# from tqdm import tqdm

# from model import TinyTransformer
# from utils import *

# device = torch.device("cpu")

# CHECKPOINT_DIR = "checkpoints"
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# def save_checkpoint(model, optimizer, epoch):

#     path = f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch}.pt"

#     torch.save({
#         "epoch": epoch,
#         "model_state": model.state_dict(),
#         "optimizer_state": optimizer.state_dict()
#     }, path)

#     print(f"\nCheckpoint saved: {path}")


# def load_latest_checkpoint(model, optimizer):

#     files = os.listdir(CHECKPOINT_DIR)

#     if not files:
#         return 0

#     files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

#     latest = files[-1]

#     checkpoint = torch.load(f"{CHECKPOINT_DIR}/{latest}", map_location=device)

#     model.load_state_dict(checkpoint["model_state"])
#     optimizer.load_state_dict(checkpoint["optimizer_state"])

#     start_epoch = checkpoint["epoch"] + 1

#     print(f"Resuming from epoch {start_epoch}")

#     return start_epoch


# def main():

#     texts = load_dataset()

#     word2idx, idx2word = build_vocab(texts)

#     sequences = build_sequences(texts, word2idx)

#     loader = get_dataloader(sequences)

#     vocab_size = len(word2idx)

#     model = TinyTransformer(vocab_size).to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     EPOCHS = 10

#     start_epoch = load_latest_checkpoint(model, optimizer)

#     for epoch in range(start_epoch, EPOCHS):

#         model.train()

#         total_loss = 0

#         progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)

#         for x, y in progress_bar:

#             x = x.to(device)
#             y = y.to(device)

#             optimizer.zero_grad()

#             logits = model(x)

#             loss = criterion(
#                 logits.view(-1, vocab_size),
#                 y.view(-1)
#             )

#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#             progress_bar.set_postfix(loss=loss.item())

#         print(f"\nEpoch {epoch+1} Total Loss {total_loss:.4f}")

#         save_checkpoint(model, optimizer, epoch+1)

#     torch.save(model.state_dict(), "model.pt")

#     print("Training complete")


# if __name__ == "__main__":
#     main()


import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from rich import print

from model import TinyTransformer
from utils import *

torch.set_num_threads(8)

console = Console()

device = torch.device("cpu")

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_checkpoint(model, optimizer, epoch):

    path = f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch}.pt"

    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, path)

    console.print(f"[green]Checkpoint saved:[/green] {path}")


def load_latest_checkpoint(model, optimizer):

    files = os.listdir(CHECKPOINT_DIR)

    if not files:
        return 0

    files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    latest = files[-1]

    checkpoint = torch.load(f"{CHECKPOINT_DIR}/{latest}", map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint["epoch"] + 1

    console.print(f"[yellow]Resuming from epoch {start_epoch}[/yellow]")

    return start_epoch


def main():

    console.print(Panel("Tiny Transformer Training", style="cyan"))

    texts = load_dataset()

    console.print("[blue]Dataset loaded[/blue]")

    word2idx, idx2word = build_vocab(texts)

    console.print(f"[blue]Vocabulary size:[/blue] {len(word2idx)}")

    sequences = build_sequences(texts, word2idx)

    loader = get_dataloader(sequences)

    vocab_size = len(word2idx)

    model = TinyTransformer(vocab_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 10

    start_epoch = load_latest_checkpoint(model, optimizer)

    for epoch in range(start_epoch, EPOCHS):

        console.print(Panel(f"Epoch {epoch+1}/{EPOCHS}", style="magenta"))

        model.train()

        total_loss = 0

        progress_bar = tqdm(
            loader,
            desc=f"Training",
            leave=True,
            ncols=100
        )

        for x, y in progress_bar:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(
                logits.view(-1, vocab_size),
                y.view(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix(loss=round(loss.item(), 4))

        console.print(f"[bold green]Epoch {epoch+1} total loss:[/bold green] {total_loss:.4f}")

        save_checkpoint(model, optimizer, epoch+1)

    torch.save(model.state_dict(), "model.pt")

    console.print(Panel("Training Complete", style="green"))


if __name__ == "__main__":
    main()
