"""
PirateNet: Physics-informed Residual Adaptive Networks

Ported from: https://github.com/PredictiveIntelligenceLab/jaxpi (pirate branch)
Paper: "PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks"
       (Wang et al., JMLR 2024)

Key idea: Modified MLP with adaptive residual connections and trainable gating.
Each PIModifiedBottleneck block has 3 dense layers with U,V gating and an alpha
parameter that blends the block output with its input (identity):
    Z = activation(Dense(x))           # dense + activation
    Z = Z * U + (1 - Z) * V           # gate with encoder outputs
    ...
    out = alpha * Z + (1 - alpha) * x  # adaptive residual

Alpha is initialized to 0.0, so the network starts as effectively shallow
(each block is an identity function), then progressively deepens during training.

Architecture scaled to ~21K params (hidden_dim=38, 4 blocks) for fair comparison
with MLP baseline (21,187 params) and TSA-PINN (21,571 params).
Original PirateNet uses hidden_dim=256 with 4 blocks (~922K params).
"""

import torch
import torch.nn as nn


class PIModifiedBottleneck(nn.Module):
    """Physics-informed Modified Bottleneck with adaptive residual gating.

    3 Dense layers with tanh activation. First two layers include U,V gating.
    Final layer blends with identity via trainable alpha parameter.
    Alpha initialized to 0.0 -> network starts as identity (effectively shallow).
    """

    def __init__(self, hidden_dim, nonlinearity=0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # Trainable alpha: controls residual vs identity blend
        self.alpha = nn.Parameter(torch.tensor(nonlinearity))

        # Xavier normal init, zero biases
        for fc in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_normal_(fc.weight)
            nn.init.zeros_(fc.bias)

    def forward(self, x, u, v):
        identity = x

        # Layer 1: Dense -> tanh -> gate with U,V
        z = torch.tanh(self.fc1(x))
        z = z * u + (1 - z) * v

        # Layer 2: Dense -> tanh -> gate with U,V
        z = torch.tanh(self.fc2(z))
        z = z * u + (1 - z) * v

        # Layer 3: Dense -> tanh
        z = torch.tanh(self.fc3(z))

        # Adaptive residual: alpha * z + (1 - alpha) * identity
        return self.alpha * z + (1 - self.alpha) * identity


class PirateNet_Cavity(nn.Module):
    """PirateNet for lid-driven cavity. Drop-in replacement for PINN_Cavity.

    Architecture: Dense projection (2->H) + U,V encoders + 4 PIModifiedBottleneck
    blocks + output layer. Uses a learned linear projection instead of Fourier
    embedding (which requires 128+ frequencies to work well; at our ~21K param
    budget with H=38, only 19 frequencies are possible, causing overfitting).

    Hidden dim: 38 (scaled down from 256 to match ~21K param budget).
    Parameters: 20,983 (vs MLP 21,187 / TSA-PINN 21,571).
    """

    def __init__(self, hidden_dim=38, num_blocks=4, output_dim=3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection: (N, 2) -> (N, hidden_dim)
        self.input_proj = nn.Linear(2, hidden_dim)
        nn.init.xavier_normal_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # Encoder U: hidden_dim -> hidden_dim
        self.encoder_u = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.encoder_u.weight)
        nn.init.zeros_(self.encoder_u.bias)

        # Encoder V: hidden_dim -> hidden_dim
        self.encoder_v = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.encoder_v.weight)
        nn.init.zeros_(self.encoder_v.bias)

        # PIModifiedBottleneck blocks
        self.blocks = nn.ModuleList([
            PIModifiedBottleneck(hidden_dim) for _ in range(num_blocks)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        # Input projection: (N, 2) -> (N, hidden_dim)
        h = torch.tanh(self.input_proj(x))

        # Encoder networks (operate on projected input)
        u = torch.tanh(self.encoder_u(h))    # (N, hidden_dim)
        v = torch.tanh(self.encoder_v(h))    # (N, hidden_dim)

        # Process through PIModifiedBottleneck blocks
        for block in self.blocks:
            h = block(h, u, v)

        # Output layer
        return self.output_layer(h)          # (N, 3)
