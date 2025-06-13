# import torch.nn as nn
# import torch

# class WifiCNN(nn.Module):
#     def __init__(self, num_ap, num_classes, num_mac):
#         super(WifiCNN, self).__init__()
#         self.embedding = nn.Embedding(
#             num_embeddings=num_mac + 1,  # padding 포함
#             embedding_dim=8,
#             padding_idx=0  # padding index
#         )
#         self.conv1 = nn.Conv1d(
#             in_channels=9, out_channels=32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(64 * num_ap, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#         self.relu = nn.ReLU()

#     def forward(self, x, mac):
#         mac_embed = self.embedding(mac)
#         x = torch.cat([x.unsqueeze(2), mac_embed], dim=2)
#         x = x.permute(0, 2, 1)
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = x.view(x.size(0), -1)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

import torch.nn as nn
import torch

# class WifiCNN(nn.Module):
#     def __init__(self, num_ap, num_classes, num_mac, dropout_rate=0.5):
#         super(WifiCNN, self).__init__()
#         self.embedding = nn.Embedding(
#             num_embeddings=num_mac + 1,  # padding 포함
#             embedding_dim=8,
#             padding_idx=0  # padding index
#         )
#         self.conv1 = nn.Conv1d(in_channels=9, out_channels=32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
#         self.dropout1 = nn.Dropout(dropout_rate)
#         self.dropout2 = nn.Dropout(dropout_rate)
#         self.fc1 = nn.Linear(64 * num_ap, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#         self.relu = nn.ReLU()

#     def forward(self, x, mac):
#         mac_embed = self.embedding(mac)  # [B, AP, 8]
#         x = torch.cat([x.unsqueeze(2), mac_embed], dim=2)  # [B, AP, 9]
#         x = x.permute(0, 2, 1)  # [B, 9, AP] -> [B, channels, AP]
#         x = self.relu(self.conv1(x))  # [B, 32, AP]
#         x = self.dropout1(x)
#         x = self.relu(self.conv2(x))  # [B, 64, AP]
#         x = self.dropout2(x)
#         x = x.view(x.size(0), -1)  # [B, 64 * AP]
#         x = self.relu(self.fc1(x))  # [B, 128]
#         x = self.fc2(x)  # [B, num_classes]
#         return x

class WifiCNN(nn.Module):
    def __init__(self, num_ap, num_classes, num_mac, dropout_rate=0.5,
                 conv1_channels=32, conv2_channels=64,
                 kernel_size=3, padding=1):
        super(WifiCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_mac + 1, embedding_dim=8, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=9, out_channels=conv1_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(conv1_channels, conv2_channels, kernel_size=kernel_size, padding=padding)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(conv2_channels * num_ap, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, mac):
        mac_embed = self.embedding(mac)  # [B, AP, 8]
        x = torch.cat([x.unsqueeze(2), mac_embed], dim=2)  # [B, AP, 9]
        x = x.permute(0, 2, 1)  # [B, channels, AP]
        x = self.relu(self.conv1(x))
        x = self.dropout1(x)
        x = self.relu(self.conv2(x))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x