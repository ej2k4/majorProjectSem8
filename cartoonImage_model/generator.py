import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, num_scenarios, num_emotions, img_channels=3):
        """
        Generator for conditional image generation
        
        Args:
            noise_dim: Dimension of noise vector
            num_scenarios: Number of scenario classes
            num_emotions: Number of emotion classes
            img_channels: Number of image channels (3 for RGB)
        """
        super(Generator, self).__init__()

        # Separate embeddings for scenario and emotion
        self.scenario_emb = nn.Embedding(num_scenarios, num_scenarios)
        self.emotion_emb = nn.Embedding(num_emotions, num_emotions)

        self.init_size = 16  # 16x16 base

        # Input: noise + scenario_embedding + emotion_embedding
        self.l1 = nn.Sequential(
            nn.Linear(noise_dim + num_scenarios + num_emotions,
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

    def forward(self, noise, scenario_labels, emotion_labels):
        """
        Generate images conditioned on scenario and emotion
        
        Args:
            noise: Noise vector of shape (batch_size, noise_dim)
            scenario_labels: Scenario class indices of shape (batch_size,)
            emotion_labels: Emotion class indices of shape (batch_size,)
            
        Returns:
            Generated images of shape (batch_size, 3, 64, 64)
        """
        scenario_input = self.scenario_emb(scenario_labels)
        emotion_input = self.emotion_emb(emotion_labels)
        
        # Concatenate noise with both embeddings
        x = torch.cat((noise, scenario_input, emotion_input), -1)
        out = self.l1(x)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img