# import torch
# import torch.nn as nn
# import torch.optim as optim
# from generator import Generator
# from discriminator import Discriminator
# from utils import get_dataloader

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Hyperparameters
# noise_dim = 100
# lr_g = 0.0002
# lr_d = 0.00005
# num_epochs = 50

# # Load data
# dataloader, num_classes = get_dataloader()

# # Models
# G = Generator(noise_dim=noise_dim, num_classes=num_classes).to(device)
# D = Discriminator(num_classes=num_classes).to(device)

# # Optimizers
# optimizer_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
# optimizer_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))

# criterion = nn.BCELoss()

# for epoch in range(num_epochs):
#     for real_imgs, labels in dataloader:

#         batch_size = real_imgs.size(0)
#         real_imgs = real_imgs.to(device)
#         labels = labels.to(device)

#         real_targets = torch.full((batch_size, 1),0.9).to(device)
#         fake_targets = torch.zeros(batch_size, 1).to(device)

#         # -------------------
#         # Train Discriminator
#         # -------------------

#         noise = torch.randn(batch_size, noise_dim).to(device)
#         fake_imgs = G(noise, labels)

#         # Add small noise to real images (stabilization trick)
#         real_imgs_noisy = real_imgs + 0.05 * torch.randn_like(real_imgs)
#         real_imgs_noisy = torch.clamp(real_imgs_noisy, -1, 1)

#         real_loss = criterion(D(real_imgs_noisy, labels), real_targets)
#         fake_loss = criterion(D(fake_imgs.detach(), labels), fake_targets)

#         d_loss = real_loss + fake_loss

#         optimizer_D.zero_grad()
#         d_loss.backward()
#         optimizer_D.step()

#         # -------------------
#         # Train Generator
#         # -------------------

#         noise = torch.randn(batch_size, noise_dim).to(device)
#         fake_imgs = G(noise, labels)

#         g_loss = criterion(D(fake_imgs, labels), real_targets)

#         optimizer_G.zero_grad()
#         g_loss.backward()
#         optimizer_G.step()

#     print(f"Epoch [{epoch+1}/{num_epochs}] "
#           f"D Loss: {d_loss.item():.4f} "
#           f"G Loss: {g_loss.item():.4f}")

# print("Training complete!")


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
# num_epochs = 30
num_epochs = 50

dataloader, num_classes = get_dataloader()

G = Generator(noise_dim=noise_dim, num_classes=num_classes).to(device)
D = Discriminator(num_classes=num_classes).to(device)

optimizer_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))

criterion = nn.BCELoss()

for epoch in range(num_epochs):
    for real_imgs, labels in dataloader:

        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        labels = labels.to(device)

        real_targets = torch.full((batch_size, 1), 0.9).to(device)
        fake_targets = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_imgs = G(noise, labels)

        real_imgs_noisy = real_imgs + 0.05 * torch.randn_like(real_imgs)
        real_imgs_noisy = torch.clamp(real_imgs_noisy, -1, 1)

        real_loss = criterion(D(real_imgs_noisy, labels), real_targets)
        fake_loss = criterion(D(fake_imgs.detach(), labels), fake_targets)

        d_loss = real_loss + fake_loss

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        # noise = torch.randn(batch_size, noise_dim).to(device)
        # fake_imgs = G(noise, labels)

        # g_loss = criterion(D(fake_imgs, labels), real_targets)

        # optimizer_G.zero_grad()
        # g_loss.backward()
        # optimizer_G.step()
        # Train Generator (twice for balance)
        for _ in range(2):
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_imgs = G(noise, labels)

            g_loss = criterion(D(fake_imgs, labels), real_targets)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"D Loss: {d_loss.item():.4f} "
          f"G Loss: {g_loss.item():.4f}")

torch.save(G.state_dict(), "generator.pth")
torch.save(D.state_dict(), "discriminator.pth")

print("Training complete!")