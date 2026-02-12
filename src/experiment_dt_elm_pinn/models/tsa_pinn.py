"""
TSA-PINN: Trainable Sinusoidal Activation PINN

Ported from: https://github.com/AmirhosseinnnKhademi/TSA-PINN
Paper: "Physics-informed neural networks with trainable sinusoidal activation functions
        for approximating the solutions of the Navier-Stokes equations"
        (Khademi, Computer Physics Communications, May 2025)

Key idea: Replace tanh activations with per-neuron trainable sinusoidal activations:
  h = 0.5 * (sin(omega * z + b) + cos(omega * z + b))
where omega is a learnable frequency parameter per neuron.

Includes Dynamic Slope Recovery (DSR) regularization to prevent gradient vanishing:
  L_reg = 1.0 / sum(exp(mean(omega_i)) for each layer i)
"""

import torch
import torch.nn as nn


class TSA_PINN_Cavity(nn.Module):
    """TSA-PINN for lid-driven cavity. Drop-in replacement for PINN_Cavity.

    Architecture: [2, 64, 64, 64, 64, 64, 64, 3] — same depth/width as MLP baseline.
    Activation: trainable sinusoidal instead of tanh.
    Extra params: 6 layers * 64 frequencies = 384 (~1.8% overhead).
    """

    def __init__(self, initial_freq=1.0, output_dim=3):
        super().__init__()
        hidden = 64
        n_hidden = 6

        # Hidden layers: weight matrices (no bias in linear — bias applied inside activation)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.freqs = nn.ParameterList()

        in_dim = 2
        for _ in range(n_hidden):
            w = nn.Parameter(torch.empty(in_dim, hidden))
            b = nn.Parameter(torch.zeros(1, hidden))
            f = nn.Parameter(torch.full((1, hidden), initial_freq))
            nn.init.xavier_normal_(w)
            self.weights.append(w)
            self.biases.append(b)
            self.freqs.append(f)
            in_dim = hidden

        # Output layer (standard linear, no activation)
        self.output_weight = nn.Parameter(torch.empty(hidden, output_dim))
        self.output_bias = nn.Parameter(torch.zeros(1, output_dim))
        nn.init.xavier_normal_(self.output_weight)

    def forward(self, x):
        h = x
        for w, b, f in zip(self.weights, self.biases, self.freqs):
            z = h @ w
            h = 0.5 * (torch.sin(f * z + b) + torch.cos(f * z + b))
        return h @ self.output_weight + self.output_bias

    def regularization_loss(self):
        """Dynamic Slope Recovery (DSR) — penalizes small frequencies."""
        reg_term = sum(torch.exp(freq.mean()) for freq in self.freqs)
        return 1.0 / reg_term
