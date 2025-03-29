# # import torch.nn as nn
# # import torch

# # class WifiCNNTransformer(nn.Module):
# #     def __init__(self, num_ap, num_classes, num_mac):
# #         super(WifiCNNTransformer, self).__init__()
# #         self.embedding = nn.Embedding(
# #             num_embeddings=num_mac + 1,  # padding 포함
# #             embedding_dim=8,
# #             padding_idx=0  # padding index
# #         )
# #         self.conv1 = nn.Conv1d(
# #             in_channels=9, out_channels=32, kernel_size=3, padding=1)
# #         self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

# #         # Transformer Encoder 추가
# #         encoder_layer = nn.TransformerEncoderLayer(
# #             d_model=64, nhead=8, dim_feedforward=128, batch_first=True
# #         )
# #         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

# #         self.fc1 = nn.Linear(64 * num_ap, 128)
# #         self.fc2 = nn.Linear(128, num_classes)
# #         self.relu = nn.ReLU()

# #     def forward(self, x, mac):
# #         mac_embed = self.embedding(mac)  # (batch, ap, embed_dim)
# #         x = torch.cat([x.unsqueeze(2), mac_embed], dim=2)  # (batch, ap, rssi+embed)
# #         x = x.permute(0, 2, 1)  # (batch, channel, ap)

# #         x = self.relu(self.conv1(x))  # (batch, 32, ap)
# #         x = self.relu(self.conv2(x))  # (batch, 64, ap)

# #         x = x.permute(0, 2, 1)  # Transformer는 (batch, seq_len, dim) → (batch, ap, 64)
# #         x = self.transformer(x)  # (batch, ap, 64)

# #         x = x.reshape(x.size(0), -1)  # flatten
# #         x = self.relu(self.fc1(x))
# #         x = self.fc2(x)
# #         return x

# import torch
# import torch.nn as nn

# class WifiCNNTransformer(nn.Module):
#     def __init__(self, num_ap, num_classes, num_mac, embedding_dim=8, transformer_heads=4, transformer_layers=2, dropout_rate=0.3):
#         super(WifiCNNTransformer, self).__init__()

#         # MAC Embedding
#         self.embedding = nn.Embedding(
#             num_embeddings=num_mac + 1,  # padding 포함
#             embedding_dim=embedding_dim,
#             padding_idx=0
#         )

#         # Conv1D Layer
#         self.conv1 = nn.Conv1d(in_channels=embedding_dim + 1, out_channels=32, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout_rate)

#         # Transformer Encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=32, nhead=transformer_heads, batch_first=True, dropout=dropout_rate
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

#         # Fully Connected Layers
#         self.fc1 = nn.Linear(32 * num_ap, 128)
#         self.fc2 = nn.Linear(128, num_classes)

#     def forward(self, x, mac):
#         mac_embed = self.embedding(mac)  # [batch, num_ap, embedding_dim]
#         x = torch.cat([x.unsqueeze(2), mac_embed], dim=2)  # [batch, num_ap, 1+embedding_dim]
#         x = x.permute(0, 2, 1)  # [batch, 1+embedding_dim, num_ap]

#         x = self.relu(self.conv1(x))  # [batch, 32, num_ap]
#         x = self.dropout(x)
#         x = x.permute(0, 2, 1)  # [batch, num_ap, 32]

#         x = self.transformer(x)  # [batch, num_ap, 32]
#         x = x.reshape(x.size(0), -1)  # Flatten

#         x = self.dropout(self.relu(self.fc1(x)))
#         x = self.fc2(x)
#         return x

import torch
import torch.nn as nn
# ---------- 모델 구조 시각화 ----------
from torchinfo import summary
from torchviz import make_dot


class WifiCNNTransformer(nn.Module):
    def __init__(
        self,
        num_ap,
        num_classes,
        num_mac,
        embedding_dim=8,
        transformer_heads=4,
        transformer_layers=2,
        dropout_rate=0.3
    ):
        super(WifiCNNTransformer, self).__init__()

        # MAC Embedding
        self.embedding = nn.Embedding(
            num_embeddings=num_mac + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # Conv1D Layer
        self.conv1 = nn.Conv1d(
            in_channels=embedding_dim + 1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, num_ap, 32))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=transformer_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers)

        # LayerNorm
        self.norm = nn.LayerNorm(32)

        # FC Layers (Bottleneck structure)
        self.fc1 = nn.Linear(32, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x, mac):
        mac_embed = self.embedding(mac)  # [B, num_ap, embedding_dim]
        # [B, num_ap, 1 + emb_dim]
        x = torch.cat([x.unsqueeze(2), mac_embed], dim=2)
        x = x.permute(0, 2, 1)  # [B, in_channels, num_ap]

        x = self.relu(self.conv1(x))  # [B, 32, num_ap]
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # [B, num_ap, 32]

        # Positional Encoding & Transformer
        x = self.transformer(x + self.positional_encoding)
        x = self.norm(x)

        # Global Average Pooling
        x = x.mean(dim=1)  # [B, 32]

        # FC Layers
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


# ---------- 실행 예제 ----------
if __name__ == "__main__":
    from torchinfo import summary
    from torchviz import make_dot
    import os

    # 가상의 파라미터
    num_ap = 100
    num_classes = 10
    num_mac = 300

    # 모델 생성
    model = WifiCNNTransformer(
        num_ap=num_ap, num_classes=num_classes, num_mac=num_mac)
    model.eval()  # 🔹 평가 모드 설정

    # 예시 입력 생성
    example_x = torch.randn(1, num_ap, dtype=torch.float32)
    example_mac = torch.randint(1, num_mac + 1, (1, num_ap), dtype=torch.long)

    print("🧠 모델 구조 요약:")
    summary(model, input_data=[example_x, example_mac])

    # 그래프 이미지 저장
    output = model(example_x, example_mac)
    dot = make_dot(output, params=dict(model.named_parameters()))
    os.makedirs("finger_printing/models", exist_ok=True)
    dot.render("finger_printing/models/model_graph",
               format="png", cleanup=True)
    print("✅ 모델 계산 그래프 이미지 저장 완료: finger_printing/model_graph.png")
