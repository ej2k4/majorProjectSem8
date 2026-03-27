import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, num_scenarios, num_emotions, img_channels=3):
        """
        Discriminator for conditional adversarial learning
        
        Args:
            num_scenarios: Number of scenario classes
            num_emotions: Number of emotion classes
            img_channels: Number of image channels (3 for RGB)
        """
        super(Discriminator, self).__init__()

        # Separate embeddings for scenario and emotion
        self.scenario_emb = nn.Embedding(num_scenarios, num_scenarios)
        self.emotion_emb = nn.Embedding(num_emotions, num_emotions)

        self.model = nn.Sequential(
            # 64x64 → 32x32
            nn.Conv2d(img_channels + num_scenarios + num_emotions, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 → 16x16
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 → 8x8
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 → 4x4
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),

            # 4x4x512 = 8192
            nn.Linear(4 * 4 * 512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, scenario_labels, emotion_labels):
        """
        Discriminate images conditioned on scenario and emotion
        
        Args:
            img: Image tensor of shape (batch_size, 3, 64, 64)
            scenario_labels: Scenario class indices of shape (batch_size,)
            emotion_labels: Emotion class indices of shape (batch_size,)
            
        Returns:
            Discriminator output (real/fake probability)
        """
        scenario_embedding = self.scenario_emb(scenario_labels)
        emotion_embedding = self.emotion_emb(emotion_labels)

        # Expand embeddings to spatial dimensions
        scenario_embedding = scenario_embedding.unsqueeze(2).unsqueeze(3)
        emotion_embedding = emotion_embedding.unsqueeze(2).unsqueeze(3)

        scenario_embedding = scenario_embedding.repeat(1, 1, img.size(2), img.size(3))
        emotion_embedding = emotion_embedding.repeat(1, 1, img.size(2), img.size(3))

        # Concatenate image with both embeddings
        d_in = torch.cat((img, scenario_embedding, emotion_embedding), 1)
        validity = self.model(d_in)
        return validity