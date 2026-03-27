"""
train.py — Training with rich terminal UI, checkpoints, early stopping, validation

Install dependency first:
    pip install rich

Usage:
    python train.py              # Start fresh or resume from latest checkpoint
    python train.py --reset      # Force restart from scratch
    python train.py --epochs 50  # Override max epochs (default: 50)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import math
import argparse
import re
import time
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, TensorDataset, random_split

# --- Rich UI imports ---
from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn,
    TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn
)
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.live import Live
from rich.rule import Rule
from rich import box

from model import TinyLSTM
from utils import preprocess_text, tokenize, build_vocab

console = Console()


# -----------------------
# Config
# -----------------------
CHECKPOINT_DIR    = "checkpoints"
CHECKPOINT_PREFIX = "checkpoint_epoch"
BEST_MODEL_PATH   = "tiny_lstm_best.pth"
FINAL_MODEL_PATH  = "tiny_lstm.pth"
VOCAB_PATH        = "vocab.json"
DATASET_PATH      = "dataset/stories.txt"
LOG_DIR           = "logs"

MAX_EPOCHS    = 50
BATCH_SIZE    = 32
SEQ_LENGTH    = 60
LEARNING_RATE = 0.001
PATIENCE      = 5
VAL_SPLIT     = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# Checkpoint Helpers
# -----------------------
def save_checkpoint(epoch, model, optimizer, train_loss, val_loss, vocab_size):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "vocab_size": vocab_size
    }, path)


def load_latest_checkpoint(model, optimizer):
    if not os.path.exists(CHECKPOINT_DIR):
        return 0
    checkpoints = sorted([
        f for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith(CHECKPOINT_PREFIX) and f.endswith(".pth")
    ])
    if not checkpoints:
        return 0
    latest = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
    ckpt = torch.load(latest, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"] + 1


# -----------------------
# Validation
# -----------------------
def evaluate(model, loader, criterion, vocab_size):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            batch_inputs  = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            output, _     = model(batch_inputs)
            loss = criterion(output.view(-1, vocab_size), batch_targets.view(-1))
            total_loss += loss.item()
    model.train()
    avg_loss   = total_loss / len(loader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


# -----------------------
# Training Log
# -----------------------
def save_training_log(history):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, "training_log.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


# -----------------------
# Qualitative Test
# -----------------------
def run_qualitative_test(model, word2idx, idx2word, scenarios, emotion="nervous"):
    os.makedirs(LOG_DIR, exist_ok=True)
    model.eval()
    lines = [
        f"Qualitative Test — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Emotion: {emotion}",
        "=" * 60
    ]
    for scenario in scenarios:
        seed  = f"<scenario_{scenario}> <emotion_{emotion}> <n> </start>"
        words = seed.lower().split()
        state = None
        for _ in range(80):
            input_ids = torch.tensor(
                [[word2idx.get(w, word2idx.get("<unk>", 0)) for w in words[-10:]]]
            ).to(device)
            output, state = model(input_ids, state)
            logits = output[0, -1]
            for word in set(words[-20:]):
                if word in word2idx:
                    logits[word2idx[word]] /= 2.0
            temperature = 0.7
            probs = torch.softmax(logits / temperature, dim=0)
            top_probs, top_indices = torch.topk(probs, 5)
            top_probs = top_probs / top_probs.sum()
            predicted_idx = top_indices[torch.multinomial(top_probs, 1)].item()
            next_word = idx2word[predicted_idx]
            if next_word == "<end>":
                break
            words.append(next_word)
        story = re.sub(r"<.*?>", "", " ".join(words))
        story = re.sub(r"\s+", " ", story).strip()
        lines.append(f"\n[{scenario}]\n{story or '(empty)'}\n" + "-" * 60)
    path = os.path.join(LOG_DIR, "qualitative_test.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    model.train()
    return path


# -----------------------
# Sparkline (mini loss chart)
# -----------------------
SPARK_CHARS = "▁▂▃▄▅▆▇█"

def sparkline(values):
    """Render a tiny bar chart from a list of floats."""
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo or 1e-9
    bars = [SPARK_CHARS[int((v - lo) / span * (len(SPARK_CHARS) - 1))] for v in values]
    return "".join(bars)


# -----------------------
# PPL colour helper
# -----------------------
def ppl_color(ppl):
    if ppl > 500:   return "red"
    if ppl > 200:   return "yellow"
    if ppl > 50:    return "cyan"
    return "green"


# -----------------------
# Build epoch summary table
# -----------------------
def build_history_table(history, patience_counter, patience):
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold white",
        expand=True
    )
    table.add_column("Epoch",       justify="right",  style="bold white",  width=7)
    table.add_column("Train Loss",  justify="right",  style="white",       width=12)
    table.add_column("Train PPL",   justify="right",  width=11)
    table.add_column("Val Loss",    justify="right",  style="white",       width=10)
    table.add_column("Val PPL",     justify="right",  width=10)
    table.add_column("Status",      justify="center", width=14)

    best_val = min((r["val_loss"] for r in history), default=float("inf"))

    for i, row in enumerate(history):
        is_best    = abs(row["val_loss"] - best_val) < 1e-6
        is_last    = i == len(history) - 1
        epoch_str  = str(row["epoch"])

        status_text = ""
        status_style = ""
        if is_best and is_last:
            status_text  = "★ best"
            status_style = "bold green"
        elif is_best:
            status_text  = "★ best"
            status_style = "dim green"
        elif is_last and patience_counter > 0:
            status_text  = f"patience {patience_counter}/{patience}"
            status_style = "yellow"

        table.add_row(
            epoch_str,
            f"{row['train_loss']:.4f}",
            Text(f"{row['train_perplexity']:.1f}", style=ppl_color(row["train_perplexity"])),
            f"{row['val_loss']:.4f}",
            Text(f"{row['val_perplexity']:.1f}",   style=ppl_color(row["val_perplexity"])),
            Text(status_text, style=status_style),
            end_section=(is_last),
        )

    return table


# -----------------------
# Build status panel
# -----------------------
def build_status_panel(epoch, max_epochs, start_epoch, history,
                       patience_counter, patience, best_val_loss,
                       elapsed_sec, device_name):
    if len(history) >= 2:
        train_spark = sparkline([r["train_loss"] for r in history])
        val_spark   = sparkline([r["val_loss"]   for r in history])
        spark_str   = f"Train [cyan]{train_spark}[/cyan]   Val [magenta]{val_spark}[/magenta]"
    else:
        spark_str = "Collecting data…"

    epochs_done  = epoch - start_epoch + 1
    epochs_total = max_epochs - start_epoch
    eta_str      = "—"
    if epochs_done > 0:
        spe      = elapsed_sec / epochs_done          # seconds per epoch
        remaining = (epochs_total - epochs_done) * spe
        eta_str  = str(timedelta(seconds=int(remaining)))

    patience_bar = ("█" * patience_counter) + ("░" * (patience - patience_counter))
    patience_col = "yellow" if patience_counter >= patience // 2 else "green"

    best_str = f"{best_val_loss:.4f}" if best_val_loss < float("inf") else "—"

    lines = [
        f"[bold]Device[/bold]       {device_name}",
        f"[bold]Epoch[/bold]        {epoch} / {max_epochs}  ({epochs_done}/{epochs_total} this run)",
        f"[bold]Best Val Loss[/bold] {best_str}",
        f"[bold]Patience[/bold]     [{patience_col}]{patience_bar}[/{patience_col}]  {patience_counter}/{patience}",
        f"[bold]ETA[/bold]          {eta_str}",
        f"[bold]Loss trend[/bold]   {spark_str}",
    ]

    return Panel(
        "\n".join(lines),
        title="[bold cyan]Training Status[/bold cyan]",
        border_style="cyan",
        expand=True
    )


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset",  action="store_true")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    MAX_EPOCHS = args.epochs

    # ── Header ──────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        "[bold cyan]TinyLSTM Story Trainer[/bold cyan]\n"
        "[dim]Social stories for autistic children · ages 5–12[/dim]",
        border_style="cyan"
    ))
    console.print()

    # ── Prep data ────────────────────────────────────────────────────────────
    with console.status("[cyan]Loading and preprocessing dataset…[/cyan]"):
        text   = preprocess_text(DATASET_PATH)
        tokens = tokenize(text)
        word2idx, idx2word = build_vocab(tokens)
        with open(VOCAB_PATH, "w") as f:
            json.dump(word2idx, f)
        vocab_size = len(word2idx)

    console.print(f"  [green]✓[/green] Vocab size    [bold]{vocab_size}[/bold] tokens")

    encoded        = [word2idx[w] for w in tokens]
    inputs, targets = [], []
    for i in range(len(encoded) - SEQ_LENGTH):
        inputs.append(encoded[i:i + SEQ_LENGTH])
        targets.append(encoded[i + 1:i + SEQ_LENGTH + 1])

    full_dataset = TensorDataset(torch.tensor(inputs), torch.tensor(targets))
    val_size     = int(len(full_dataset) * VAL_SPLIT)
    train_size   = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    console.print(f"  [green]✓[/green] Train samples [bold]{train_size}[/bold]  |  Val samples [bold]{val_size}[/bold]")
    console.print(f"  [green]✓[/green] Device        [bold]{device}[/bold]")
    console.print()

    # ── Model ────────────────────────────────────────────────────────────────
    model     = TinyLSTM(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_epoch = 0
    if not args.reset:
        start_epoch = load_latest_checkpoint(model, optimizer)
        if start_epoch > 0:
            console.print(f"  [yellow]⟳[/yellow]  Resuming from epoch [bold]{start_epoch}[/bold]\n")
    else:
        console.print("  [yellow]⚠[/yellow]  --reset: starting from scratch\n")

    # ── Early stopping state ──────────────────────────────────────────────────
    best_val_loss   = float("inf")
    patience_counter = 0
    history         = []
    train_start     = time.time()
    stopped_early   = False

    console.print(Rule("[bold]Training[/bold]"))
    console.print()

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, MAX_EPOCHS):

        model.train()
        total_train_loss = 0
        batches          = len(train_loader)
        epoch_start      = time.time()

        # -- Batch progress bar --
        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn(f"  [bold]Epoch {epoch}/{MAX_EPOCHS}[/bold]"),
            BarColumn(bar_width=30, style="cyan", complete_style="bold cyan"),
            MofNCompleteColumn(),
            TextColumn("[dim]loss:[/dim] {task.fields[loss]}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,          # disappears when done, replaced by summary
        ) as progress:
            task = progress.add_task("batch", total=batches, loss="—")

            for batch_inputs, batch_targets in train_loader:
                batch_inputs  = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)

                optimizer.zero_grad()
                output, _ = model(batch_inputs)
                loss = criterion(
                    output.view(-1, vocab_size),
                    batch_targets.view(-1)
                )
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                progress.update(task, advance=1, loss=f"{loss.item():.4f}")

        # -- Validation --
        avg_train_loss  = total_train_loss / batches
        val_loss, val_ppl = evaluate(model, val_loader, criterion, vocab_size)
        train_ppl       = math.exp(avg_train_loss)
        epoch_time      = time.time() - epoch_start

        history.append({
            "epoch":            epoch,
            "train_loss":       round(avg_train_loss, 4),
            "train_perplexity": round(train_ppl, 2),
            "val_loss":         round(val_loss,  4),
            "val_perplexity":   round(val_ppl,   2),
        })

        save_checkpoint(epoch, model, optimizer, avg_train_loss, val_loss, vocab_size)

        # -- Early stopping check --
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(), "vocab_size": vocab_size}, BEST_MODEL_PATH)

        else:
            patience_counter += 1

        # -- Print epoch summary row --
        status_icon  = "[bold green]★ best[/bold green]" if improved else f"[yellow]patience {patience_counter}/{PATIENCE}[/yellow]"
        train_ppl_str = f"[{ppl_color(train_ppl)}]{train_ppl:.1f}[/{ppl_color(train_ppl)}]"
        val_ppl_str   = f"[{ppl_color(val_ppl)}]{val_ppl:.1f}[/{ppl_color(val_ppl)}]"

        console.print(
            f"  Epoch [bold]{epoch:>3}[/bold]  "
            f"train [white]{avg_train_loss:.4f}[/white] (PPL {train_ppl_str})  "
            f"val [white]{val_loss:.4f}[/white] (PPL {val_ppl_str})  "
            f"[dim]{epoch_time:.1f}s[/dim]  {status_icon}"
        )

        if patience_counter >= PATIENCE:
            console.print()
            console.print(Panel(
                f"[yellow]No improvement for [bold]{PATIENCE}[/bold] epochs.\n"
                f"Stopping at epoch [bold]{epoch}[/bold]. "
                f"Best val loss: [bold green]{best_val_loss:.4f}[/bold green][/yellow]",
                title="[yellow]Early Stopping[/yellow]",
                border_style="yellow"
            ))
            stopped_early = True
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    total_time = time.time() - train_start

    console.print()
    console.print(Rule("[bold]Results[/bold]"))
    console.print()
    console.print(build_history_table(history, patience_counter, PATIENCE))
    console.print()

    # Loss sparkline summary
    if len(history) > 1:
        t_spark = sparkline([r["train_loss"] for r in history])
        v_spark = sparkline([r["val_loss"]   for r in history])
        console.print(f"  Train loss [cyan]{t_spark}[/cyan]  (left=early, right=late)")
        console.print(f"  Val loss   [magenta]{v_spark}[/magenta]")
        console.print()

    # Save artefacts
    torch.save({"model_state_dict": model.state_dict(), "vocab_size": vocab_size}, FINAL_MODEL_PATH)
    save_training_log(history)

    # Qualitative test
    console.print(Rule("[bold]Qualitative Test[/bold]"))
    console.print()
    sample_scenarios = ["dentist_visit", "haircut", "school_bus", "birthday_party", "bedtime", "classroom_test"]

    with console.status("[cyan]Generating sample stories…[/cyan]"):
        qpath = run_qualitative_test(model, word2idx, idx2word, sample_scenarios)

    # Done panel
    epochs_run = len(history)
    stop_reason = "early stopping" if stopped_early else f"reached epoch {MAX_EPOCHS}"

    console.print(Panel(
        f"[green]Training complete[/green] — {stop_reason}\n\n"
        f"  Epochs trained   [bold]{epochs_run}[/bold]\n"
        f"  Best val loss    [bold green]{best_val_loss:.4f}[/bold green]\n"
        f"  Total time       [bold]{timedelta(seconds=int(total_time))}[/bold]\n\n"
        f"  [cyan]tiny_lstm_best.pth[/cyan]  ← use this for inference\n"
        f"  [dim]tiny_lstm.pth[/dim]       ← last epoch fallback\n"
        f"  [dim]logs/training_log.json[/dim]\n"
        f"  [dim]{qpath}[/dim]",
        title="[bold green]Done[/bold green]",
        border_style="green"
    ))
    console.print()