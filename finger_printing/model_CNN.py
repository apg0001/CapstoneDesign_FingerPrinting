import torch.nn as nn
import torch

class WifiCNN(nn.Module):
    def __init__(self, num_ap, num_classes, num_mac):
        super(WifiCNN, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_mac + 1,  # padding 포함
            embedding_dim=8,
            padding_idx=0  # padding index
        )
        self.conv1 = nn.Conv1d(
            in_channels=9, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * num_ap, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, mac):
        mac_embed = self.embedding(mac)
        x = torch.cat([x.unsqueeze(2), mac_embed], dim=2)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x