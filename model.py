"""Small residual MLP teacher with the fuzzy classifier head."""

import torch
from torch import nn
from fuzzy_head import FuzzyLayer, DefuzzyLinearLayer


class DenseBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.05):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size), nn.BatchNorm1d(output_size), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, size, dropout=0.05):
        super().__init__()
        self.block = DenseBlock(size, size, dropout)

    def forward(self, x):
        return x + self.block(x)


class FuzzyTeacher(nn.Module):
    def __init__(self, input_size, hidden_size=16, fuzzy_rules=8, n_classes=2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            DenseBlock(input_size, hidden_size), ResidualBlock(hidden_size), ResidualBlock(hidden_size)
        )
        self.fuzzy_layer = FuzzyLayer.from_dimensions(hidden_size, fuzzy_rules)
        self.pred_layer = DefuzzyLinearLayer.from_dimensions(fuzzy_rules, n_classes)

    def forward(self, x):
        return self.pred_layer(self.fuzzy_layer(self.feature_extractor(x)))

    def saliency_gradient(self, x, target=None):
        """Eq. 3 standard-gradient saliency, ``|dY / dx|``.

        ``Y`` is the selected defuzzified model output. The returned gradients
        are in the units of ``x``; in ``example.py`` that is per training-set
        standard deviation. Call this method after ``eval()``.
        """
        x = x.detach().clone().requires_grad_(True)
        output = self(x)
        if target is None:
            target = output.argmax(1)
        target = torch.as_tensor(target, device=output.device, dtype=torch.long)
        if target.ndim == 0:
            target = target.expand(output.shape[0])
        if target.shape != (output.shape[0],):
            raise ValueError("target must be a scalar or one class index per sample")
        gradient, = torch.autograd.grad(output.gather(1, target[:, None]).sum(), x)
        return gradient.abs()

    def firing_strength(self, x):
        """Total rule activation per sample.

        Above the normalization epsilon, relative activations determine the
        output; a dominant rule makes it approach that rule's consequences.
        Only when total activation is negligible relative to the epsilon does
        the output approach zero. This diagnostic is not a validated
        out-of-distribution detector.
        """
        return self.fuzzy_layer(self.feature_extractor(x)).sum(-1)
