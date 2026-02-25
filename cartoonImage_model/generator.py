# import torch
# import torch.nn as nn

# class Generator(nn.Module):
#     def __init__(self, noise_dim, num_classes, img_channels=3):
#         super(Generator, self).__init__()

#         self.label_emb = nn.Embedding(num_classes, num_classes)

#         self.init_size = 8  # start from 8x8
#         self.l1 = nn.Sequential(
#             nn.Linear(noise_dim + num_classes, 256 * self.init_size * self.init_size)
#         )

#         self.conv_blocks = nn.Sequential(
#             nn.BatchNorm2d(256),

#             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(64, img_channels, 3, stride=1, padding=1),
#             nn.Tanh()
#         )

#     def forward(self, noise, labels):
#         label_input = self.label_emb(labels)
#         x = torch.cat((noise, label_input), -1)
#         out = self.l1(x)
#         out = out.view(out.shape[0], 256, self.init_size, self.init_size)
#         img = self.conv_blocks(out)
#         return img


# import torch
# import torch.nn as nn

# class Generator(nn.Module):
#     def __init__(self, noise_dim, num_classes, img_channels=3):
#         super(Generator, self).__init__()

#         self.label_emb = nn.Embedding(num_classes, num_classes)

#         self.init_size = 16  # 16x16 → upsample to 64x64

#         self.l1 = nn.Sequential(
#             nn.Linear(noise_dim + num_classes,
#                       256 * self.init_size * self.init_size)
#         )

#         self.conv_blocks = nn.Sequential(
#             nn.BatchNorm2d(256),

#             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 16 → 32
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 32 → 64
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(64, img_channels, 3, stride=1, padding=1),
#             nn.Tanh()
#         )

#     def forward(self, noise, labels):
#         label_input = self.label_emb(labels)
#         x = torch.cat((noise, label_input), -1)
#         out = self.l1(x)
#         out = out.view(out.shape[0], 256, self.init_size, self.init_size)
#         img = self.conv_blocks(out)
#         return img


import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, img_channels=3):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.init_size = 16  # 16x16 base

        self.l1 = nn.Sequential(
            nn.Linear(noise_dim + num_classes,
                      256 * self.init_size * self.init_size)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        x = torch.cat((noise, label_input), -1)
        out = self.l1(x)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img