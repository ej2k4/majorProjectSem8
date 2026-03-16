import torch
import torch.nn as nn
from tqdm import tqdm
import os

from model import TinyTransformer
from utils import (
    load_dataset,
    build_vocab,
    save_vocab,
    create_sequences,
    get_dataloader,
    save_model,
    save_json_log
)


def main():

    # -----------------------
    # System setup
    # -----------------------

    torch.set_num_threads(os.cpu_count())

    device = torch.device("cpu")

    os.makedirs("checkpoints", exist_ok=True)

    # -----------------------
    # Load dataset
    # -----------------------

    texts = load_dataset()

    # -----------------------
    # Build vocabulary
    # -----------------------

    vocab = build_vocab(texts)

    save_vocab(vocab)

    print("Vocabulary size:", vocab.size)

    # -----------------------
    # Create training sequences
    # -----------------------

    seq_len = 20

    sequences = create_sequences(texts, vocab, seq_len)

    print("Total training sequences:", len(sequences))

    loader = get_dataloader(sequences, batch_size=16)

    # -----------------------
    # Model
    # -----------------------

    model = TinyTransformer(
        vocab_size=vocab.size,
        embed_size=128,
        num_layers=2,
        heads=4
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 5

    # -----------------------
    # Resume training
    # -----------------------

    start_epoch = 0

    latest_checkpoint = "checkpoints/latest_model.pt"

    if os.path.exists(latest_checkpoint):

        print("Found existing checkpoint. Resuming training...")

        checkpoint = torch.load(latest_checkpoint, map_location=device)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        start_epoch = checkpoint["epoch"] + 1

        print(f"Resuming from epoch {start_epoch}")

    # -----------------------
    # Training loop
    # -----------------------

    for epoch in range(start_epoch, epochs):

        total_loss = 0

        loop = tqdm(loader)

        for batch in loop:

            # dataloader returns tuple
            x = batch[0].to(device)

            optimizer.zero_grad()

            logits = model(x)

            # Teacher forcing: predict next token
            input_logits = logits[:, :-1, :].contiguous()
            target_tokens = x[:, 1:].contiguous()

            loss = criterion(
                input_logits.view(-1, input_logits.size(-1)),
                target_tokens.view(-1)
            )

            loss.backward()

            # Gradient clipping for transformer stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

        epoch_loss = total_loss / len(loader)

        print("Epoch Loss:", epoch_loss)

        # -----------------------
        # Save training log
        # -----------------------

        save_json_log({
            "epoch": epoch + 1,
            "loss": epoch_loss
        })

        # -----------------------
        # Save checkpoint
        # -----------------------

        checkpoint_data = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }

        torch.save(
            checkpoint_data,
            f"checkpoints/model_epoch_{epoch+1}.pt"
        )

        torch.save(
            checkpoint_data,
            "checkpoints/latest_model.pt"
        )

        print(f"Checkpoint saved for epoch {epoch+1}")

    print("\nTraining complete")

    # -----------------------
    # Save final model
    # -----------------------

    save_model(model, "final_model.pt")

    print("Final model saved")


if __name__ == "__main__":
    main()