"""
2-player Yahtzee neural network.

V(cur_board, opp_board, diff) = P(current player wins | both play optimally).

Input:  36-dim feature vector (see yahtzee_2player_env.extract_features)
Output: scalar in (0, 1)

Symmetry invariant enforced at training time, not architecture level:
  V(cur, opp, diff) = 1 - V(opp, cur, -diff)
"""

import numpy as np
import torch
import torch.nn as nn


FEATURE_DIM = 36

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class YahtzeeValueNet(nn.Module):
    def __init__(self, hidden=256, layers=3):
        super().__init__()
        blocks = []
        in_dim = FEATURE_DIM
        for _ in range(layers):
            blocks += [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            in_dim = hidden
        blocks += [nn.Linear(hidden, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x).squeeze(-1)   # (N,) or scalar


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

class YahtzeeInference:
    """
    Thin wrapper for batch inference during tree search.
    Keeps the model on device and handles numpy↔tensor conversion.
    """
    def __init__(self, model, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, features_np):
        """features_np: (N, 36) float32 → (N,) float32 numpy."""
        x = torch.from_numpy(features_np).to(self.device)
        return self.model(x).cpu().numpy()

    @torch.no_grad()
    def predict_single(self, feat):
        """feat: (36,) float32 → scalar float."""
        x = torch.from_numpy(feat[None]).to(self.device)
        return float(self.model(x))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    @classmethod
    def load(cls, path, hidden=256, layers=3, device=None):
        model = YahtzeeValueNet(hidden=hidden, layers=layers)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        return cls(model, device=device)


# ---------------------------------------------------------------------------
# EMA target network
# ---------------------------------------------------------------------------

class EMANet:
    """Exponential moving average of model weights for stable TD targets."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.named_parameters():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply_to(self, model):
        """Copy EMA weights into model (in-place)."""
        for k, v in model.named_parameters():
            v.data.copy_(self.shadow[k])

    def get_model(self, hidden=256, layers=3):
        """Return a new model with EMA weights loaded."""
        m = YahtzeeValueNet(hidden=hidden, layers=layers)
        for k, v in m.named_parameters():
            v.data.copy_(self.shadow[k])
        return m
