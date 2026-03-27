"""
import torch
import torch.nn as nn
import torch.optim as optim
from generator import Generator
from discriminator import Discriminator
from utils import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

noise_dim = 100
lr_g = 0.0002
lr_d = 0.00002
num_epochs = 50

# Load data and get class information
dataloader, num_scenarios, num_emotions = get_dataloader()

print(f"Number of scenarios: {num_scenarios}")
print(f"Number of emotions: {num_emotions}")
print(f"Total classes (scenario-emotion combos): {num_scenarios * num_emotions}")

# Initialize models
G = Generator(noise_dim=noise_dim, num_scenarios=num_scenarios, num_emotions=num_emotions).to(device)
D = Discriminator(num_scenarios=num_scenarios, num_emotions=num_emotions).to(device)

# Initialize optimizers
optimizer_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))

criterion = nn.BCELoss()

print("Starting training...")

for epoch in range(num_epochs):
    total_d_loss = 0
    total_g_loss = 0
    num_batches = 0

    for real_imgs, scenario_labels, emotion_labels in dataloader:

        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        scenario_labels = scenario_labels.to(device)
        emotion_labels = emotion_labels.to(device)

        real_targets = torch.full((batch_size, 1), 0.9).to(device)
        fake_targets = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_imgs = G(noise, scenario_labels, emotion_labels)

        # Add small noise to real images (stabilization trick)
        real_imgs_noisy = real_imgs + 0.05 * torch.randn_like(real_imgs)
        real_imgs_noisy = torch.clamp(real_imgs_noisy, -1, 1)

        real_loss = criterion(D(real_imgs_noisy, scenario_labels, emotion_labels), real_targets)
        fake_loss = criterion(D(fake_imgs.detach(), scenario_labels, emotion_labels), fake_targets)

        d_loss = real_loss + fake_loss

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Train Generator (twice for balance)
        for _ in range(2):
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_imgs = G(noise, scenario_labels, emotion_labels)

            g_loss = criterion(D(fake_imgs, scenario_labels, emotion_labels), real_targets)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

        total_d_loss += d_loss.item()
        total_g_loss += g_loss.item()
        num_batches += 1

    avg_d_loss = total_d_loss / num_batches
    avg_g_loss = total_g_loss / num_batches

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"D Loss: {avg_d_loss:.4f} "
          f"G Loss: {avg_g_loss:.4f}")

# Save trained models
torch.save(G.state_dict(), "generator.pth")
torch.save(D.state_dict(), "discriminator.pth")

print("Training complete! Models saved as generator.pth and discriminator.pth")
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
import torchvision.utils as vutils

from generator import Generator
from discriminator import Discriminator
from utils import get_dataloader

# -------------------------
# Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
console = Console()

noise_dim = 100
lr_g = 0.0002
lr_d = 0.00002
num_epochs = 50
checkpoint_dir = "checkpoints"
sample_dir = "samples"

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)

console.print(f"[bold green]Using device: {device}[/bold green]")

# -------------------------
# Load Data
# -------------------------
dataloader, num_scenarios, num_emotions = get_dataloader()

console.print(f"[cyan]Scenarios:[/cyan] {num_scenarios}")
console.print(f"[cyan]Emotions:[/cyan] {num_emotions}")
console.print(f"[cyan]Total Classes:[/cyan] {num_scenarios * num_emotions}\n")

# -------------------------
# Models
# -------------------------
G = Generator(noise_dim, num_scenarios, num_emotions).to(device)
D = Discriminator(num_scenarios, num_emotions).to(device)

optimizer_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))

criterion = nn.BCELoss()

# -------------------------
# Checkpoint Functions
# -------------------------
def save_checkpoint(epoch, G, D, optimizer_G, optimizer_D, d_loss, g_loss):
    path = os.path.join(checkpoint_dir, f"gan_epoch_{epoch}.pt")

    torch.save({
        "epoch": epoch,
        "G_state": G.state_dict(),
        "D_state": D.state_dict(),
        "opt_G": optimizer_G.state_dict(),
        "opt_D": optimizer_D.state_dict(),
        "d_loss": d_loss,
        "g_loss": g_loss
    }, path)

    console.print(f"[green]✅ Checkpoint saved:[/green] {path}")


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=device)

    G.load_state_dict(checkpoint["G_state"])
    D.load_state_dict(checkpoint["D_state"])
    optimizer_G.load_state_dict(checkpoint["opt_G"])
    optimizer_D.load_state_dict(checkpoint["opt_D"])

    console.print(f"[yellow]🔁 Resumed from epoch {checkpoint['epoch']}[/yellow]")

    return checkpoint["epoch"]


# -------------------------
# Resume Option
# -------------------------
start_epoch = 0
resume_path = ""  # 👉 put path like "checkpoints/gan_epoch_20.pt"

if resume_path and os.path.exists(resume_path):
    start_epoch = load_checkpoint(resume_path)

# -------------------------
# Training
# -------------------------
console.print("\n[bold green]🚀 Starting Training...[/bold green]\n")

best_g_loss = float("inf")

for epoch in range(start_epoch, num_epochs):

    total_d_loss = 0
    total_g_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for real_imgs, scenario_labels, emotion_labels in progress_bar:

        batch_size = real_imgs.size(0)

        real_imgs = real_imgs.to(device)
        scenario_labels = scenario_labels.to(device)
        emotion_labels = emotion_labels.to(device)

        real_targets = torch.full((batch_size, 1), 0.9).to(device)
        fake_targets = torch.zeros(batch_size, 1).to(device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_imgs = G(noise, scenario_labels, emotion_labels)

        real_imgs_noisy = real_imgs + 0.05 * torch.randn_like(real_imgs)
        real_imgs_noisy = torch.clamp(real_imgs_noisy, -1, 1)

        real_loss = criterion(D(real_imgs_noisy, scenario_labels, emotion_labels), real_targets)
        fake_loss = criterion(D(fake_imgs.detach(), scenario_labels, emotion_labels), fake_targets)

        d_loss = real_loss + fake_loss

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        # Train Generator
        # ---------------------
        for _ in range(2):
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_imgs = G(noise, scenario_labels, emotion_labels)

            g_loss = criterion(D(fake_imgs, scenario_labels, emotion_labels), real_targets)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

        total_d_loss += d_loss.item()
        total_g_loss += g_loss.item()
        num_batches += 1

        progress_bar.set_postfix({
            "D_loss": f"{d_loss.item():.3f}",
            "G_loss": f"{g_loss.item():.3f}"
        })

    avg_d_loss = total_d_loss / num_batches
    avg_g_loss = total_g_loss / num_batches

    # -------------------------
    # Epoch Summary
    # -------------------------
    table = Table(title=f"Epoch {epoch+1} Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Discriminator Loss", f"{avg_d_loss:.4f}")
    table.add_row("Generator Loss", f"{avg_g_loss:.4f}")

    console.print(table)

    # -------------------------
    # Collapse Detection
    # -------------------------
    if avg_g_loss < 0.1:
        console.print("[red]⚠️ Possible Generator Collapse[/red]")

    if avg_d_loss < 0.1:
        console.print("[red]⚠️ Discriminator too strong[/red]")

    if avg_d_loss > 2.0:
        console.print("[red]⚠️ Discriminator too weak[/red]")

    # -------------------------
    # Save Best Model
    # -------------------------
    if avg_g_loss < best_g_loss:
        best_g_loss = avg_g_loss
        torch.save(G.state_dict(), "best_generator.pth")
        console.print("[bold yellow]🏆 Best Generator Saved![/bold yellow]")

    # -------------------------
    # Save Checkpoint
    # -------------------------
    if (epoch + 1) % 5 == 0:
        save_checkpoint(epoch + 1, G, D, optimizer_G, optimizer_D, avg_d_loss, avg_g_loss)

    # -------------------------
    # Save Samples
    # -------------------------
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            sample_noise = torch.randn(16, noise_dim).to(device)
            sample_imgs = G(sample_noise, scenario_labels[:16], emotion_labels[:16])

            vutils.save_image(
                sample_imgs,
                os.path.join(sample_dir, f"epoch_{epoch+1}.png"),
                normalize=True
            )

    # -------------------------
    # Log to File
    # -------------------------
    with open("training_log.txt", "a") as f:
        f.write(f"{epoch+1},{avg_d_loss},{avg_g_loss}\n")

# -------------------------
# Final Save
# -------------------------
torch.save(G.state_dict(), "generator_final.pth")
torch.save(D.state_dict(), "discriminator_final.pth")

console.print("\n[bold green]✅ Training Complete![/bold green]")