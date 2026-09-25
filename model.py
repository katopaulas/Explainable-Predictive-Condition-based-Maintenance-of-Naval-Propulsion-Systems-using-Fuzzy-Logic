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

    def saliency_probability(self, x, target=None):
        """Eq. 3: S_i(x) = |d p_T(y_l | x) / d x_i|.

        p_T is the softmax over the defuzzified outputs of Eq. 2, not the raw
        output itself. Returns one row per sample, in the units of x: with the
        standardised pipeline of example.py that is "per standard deviation of
        that feature", which is what makes the components comparable.

        Requires eval() so batch normalisation does not couple samples.
        """
        x = x.detach().clone().requires_grad_(True)
        p = torch.softmax(self(x), dim=1)
        if target is None:
            target = p.argmax(1)
        target = torch.as_tensor(target).reshape(-1, 1)
        p.gather(1, target).sum().backward()
        return x.grad.abs()

    def saliency_rule_activation(self, x, k=3):
        """Eq. 4: S~_i(x) = sum_{c in F_k} |d mu_c(f(x)) / d x_i|.

        F_k is the set of k fuzzy rules with the highest membership mu_c(f(x)).
        Gradients are taken with respect to the input x, so they propagate back
        through the feature extractor f; same units as `saliency_probability`.

        Requires eval() so batch normalisation does not couple samples.
        """
        x = x.detach().clone().requires_grad_(True)
        mu = self.fuzzy_layer(self.feature_extractor(x))
        top = mu.topk(min(k, mu.shape[1]), dim=1).indices
        saliency = torch.zeros_like(x)
        for j in range(top.shape[1]):
            grad, = torch.autograd.grad(mu.gather(1, top[:, j:j + 1]).sum(), x,
                                        retain_graph=True)
            saliency = saliency + grad.abs()
        return saliency.detach()

    def firing_strength(self, x):
        """Total rule activation per sample.

        The normalisation inside the head divides this away, so a point far from
        every rule still produces finite-looking logits (they collapse towards 0
        and argmax then returns an arbitrary class). Use this as the
        out-of-distribution / abstention signal.
        """
        return self.fuzzy_layer(self.feature_extractor(x)).sum(-1)
