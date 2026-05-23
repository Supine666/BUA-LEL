import torch.nn as nn

class EnhancedClassifier(nn.Module):
    def __init__(self, feature_dim, output_size=3, hidden_dims=(512, 256), dropout_rate=0.3):
        super().__init__()
        layers = []
        prev = feature_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ]
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)