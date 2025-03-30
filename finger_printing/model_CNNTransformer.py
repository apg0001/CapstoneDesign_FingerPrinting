import torch
import torch.nn as nn

class WifiCNNTransformer(nn.Module):
    def __init__(self, num_ap, num_classes, num_mac, embedding_dim=16, transformer_heads=4, transformer_layers=4, dropout_rate=0.3):
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

        self.norm = nn.LayerNorm(32)

        # Fully Connected Layers
        self.fc1 = nn.Linear(32 * num_ap, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x, mac):
        mac_embed = self.embedding(mac)  # [batch, num_ap, embedding_dim]
        x = torch.cat([x.unsqueeze(2), mac_embed], dim=2)  # [batch, num_ap, 1+embedding_dim]
        x = x.permute(0, 2, 1)  # [batch, 1+embedding_dim, num_ap]

        x = self.relu(self.conv1(x))  # [batch, 32, num_ap]
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # [batch, num_ap, 32]

        x = self.transformer(x)
        x = self.norm(x)
        x = x.reshape(x.size(0), -1)  # Flatten

        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary
    import matplotlib.pyplot as plt
    from torchviz import make_dot

    model = WifiCNNTransformer(num_ap=100, num_classes=10, num_mac=300)
    model.eval()
    print("\U0001F9E0 모델 구조 요약:")
    summary(model, input_data=(torch.zeros(1, 100), torch.zeros(1, 100, dtype=torch.long)))

    example_x = torch.randn(1, 100)
    example_mac = torch.randint(0, 300, (1, 100))
    output = model(example_x, example_mac)

    make_dot(output, params=dict(model.named_parameters())).render("model_architecture", format="png")
    print("🖼️ 모델 구조 이미지 저장 완료: model_architecture.png")