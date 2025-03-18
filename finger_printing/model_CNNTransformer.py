# import torch.nn as nn
# import torch

# class WifiCNNTransformer(nn.Module):
#     def __init__(self, num_ap, num_classes, num_mac):
#         super(WifiCNNTransformer, self).__init__()
#         self.embedding = nn.Embedding(
#             num_embeddings=num_mac + 1,  # padding 포함
#             embedding_dim=8,
#             padding_idx=0  # padding index
#         )
#         self.conv1 = nn.Conv1d(
#             in_channels=9, out_channels=32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        
#         # Transformer Encoder 추가
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=64, nhead=8, dim_feedforward=128, batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

#         self.fc1 = nn.Linear(64 * num_ap, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#         self.relu = nn.ReLU()

#     def forward(self, x, mac):
#         mac_embed = self.embedding(mac)  # (batch, ap, embed_dim)
#         x = torch.cat([x.unsqueeze(2), mac_embed], dim=2)  # (batch, ap, rssi+embed)
#         x = x.permute(0, 2, 1)  # (batch, channel, ap)

#         x = self.relu(self.conv1(x))  # (batch, 32, ap)
#         x = self.relu(self.conv2(x))  # (batch, 64, ap)

#         x = x.permute(0, 2, 1)  # Transformer는 (batch, seq_len, dim) → (batch, ap, 64)
#         x = self.transformer(x)  # (batch, ap, 64)

#         x = x.reshape(x.size(0), -1)  # flatten
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

import torch
import torch.nn as nn

class WifiCNNTransformer(nn.Module):
    def __init__(self, num_ap, num_classes, num_mac, embedding_dim=8, transformer_heads=4, transformer_layers=2, dropout_rate=0.3):
        super(WifiCNNTransformer, self).__init__()
        
        # MAC Embedding
        self.embedding = nn.Embedding(
            num_embeddings=num_mac + 1,  # padding 포함
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # Conv1D Layer
        self.conv1 = nn.Conv1d(in_channels=embedding_dim + 1, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=transformer_heads, batch_first=True, dropout=dropout_rate
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(32 * num_ap, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x, mac):
        mac_embed = self.embedding(mac)  # [batch, num_ap, embedding_dim]
        x = torch.cat([x.unsqueeze(2), mac_embed], dim=2)  # [batch, num_ap, 1+embedding_dim]
        x = x.permute(0, 2, 1)  # [batch, 1+embedding_dim, num_ap]
        
        x = self.relu(self.conv1(x))  # [batch, 32, num_ap]
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # [batch, num_ap, 32]
        
        x = self.transformer(x)  # [batch, num_ap, 32]
        x = x.reshape(x.size(0), -1)  # Flatten
        
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x