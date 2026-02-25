import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, num_classes, img_channels=3):
        super(Discriminator, self).__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Conv2d(img_channels + 1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_map = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        label_map = label_map.repeat(1, 1, img.size(2), img.size(3))
        d_in = torch.cat((img, label_map[:, :1, :, :]), 1)
        validity = self.model(d_in)
        return validity
