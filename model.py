"""Small residual MLP teacher with the fuzzy classifier head."""

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
