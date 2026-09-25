"""Differentiable fuzzy classifier layers used by the reference teacher.

Membership is exp(-||A(x - c)||), an exponential (Laplace-type) kernel -- not
exp(-||A(x - c)||^2 / 2). `rots` parameterises the off-diagonals of the
symmetric shape matrix A; it is not a rotation and A is not constrained to be
non-singular, so a rule can degenerate into a ridge of membership 1.
"""

import numpy as np
import torch
from torch import nn, Tensor


class FuzzyLayer(torch.nn.Module):
    def __init__(self, initial_centers, initial_scales, trainable=True):
        super().__init__()
        if np.shape(initial_centers) != np.shape(initial_scales):
            raise ValueError("initial_centers shape does not match initial_scales")

        self.size_out, self.size_in = np.shape(initial_centers)
        const_row = np.zeros(self.size_in + 1)
        const_row[self.size_in] = 1
        const_row = np.array([const_row] * self.size_out).reshape(
            self.size_out, 1, self.size_in + 1
        )
        self.c_r = nn.Parameter(torch.FloatTensor(const_row), requires_grad=False)
        self.c_one = nn.Parameter(torch.FloatTensor([1]), requires_grad=False)
        self.scales = nn.Parameter(initial_scales, requires_grad=trainable)
        self.rots = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.size_out, self.size_in - i - 1), requires_grad=trainable)
             for i in range(self.size_in - 1)]
        )
        self.centroids = nn.Parameter(
            initial_centers.reshape(self.size_out, self.size_in, 1), requires_grad=trainable
        )

    @classmethod
    def from_dimensions(cls, size_in, size_out, trainable=True):
        return cls(torch.randn(size_out, size_in), torch.ones(size_out, size_in), trainable)

    @classmethod
    def from_centers(cls, initial_centers, trainable=True):
        initial_centers = torch.FloatTensor(np.multiply(-1, initial_centers))
        initial_scales = torch.ones_like(initial_centers)
        return cls(initial_centers, initial_scales, trainable)

    @classmethod
    def from_centers_and_scales(cls, initial_centers, initial_scales, trainable=True):
        initial_centers = torch.FloatTensor(np.multiply(-1, initial_centers))
        initial_scales = torch.FloatTensor(initial_scales)
        return cls(initial_centers, initial_scales, trainable)

    def get_scales_and_rot(self):
        A = torch.diag_embed(self.scales)
        for i, r in enumerate(self.rots):
            A = A + torch.diag_embed(r, i + 1) + torch.diag_embed(r, i + 1, -1, -2)
        return A

    def forward(self, input: Tensor) -> Tensor:
        """Returns exp(-||A(x - c)||) per rule; 1 at the centre, decaying outward."""
        batch_size = input.shape[0]
        A = torch.cat((self.get_scales_and_rot(), self.centroids), dim=2)
        transform = torch.cat((A, self.c_r), dim=1)
        extended_input = torch.cat((input, self.c_one.repeat(batch_size, 1)), dim=1).T
        distance = torch.norm(torch.matmul(transform, extended_input)[:, :self.size_in], p=2, dim=1)
        return torch.exp(-distance).T

    def set_requires_grad_rot(self, requires_grad):
        for i in range(self.size_in - 1):
            self.rots[i].requires_grad = requires_grad

    def set_requires_grad_scales(self, requires_grad):
        self.scales.requires_grad = requires_grad

    def set_requires_grad_centroids(self, requires_grad):
        self.centroids.requires_grad = requires_grad

    def get_centroids(self):
        lh = self.get_scales_and_rot()
        rh = self.centroids.squeeze(-1)
        return torch.linalg.solve(lh, -rh)

    def get_transformation_matrix_eigenvals(self):
        return torch.linalg.eigvals(self.get_scales_and_rot())

    def get_transformation_matrix(self):
        A = torch.cat((self.get_scales_and_rot(), self.centroids), 2)
        return torch.cat([A, self.c_r], 1)


class DefuzzyLinearLayer(torch.nn.Module):
    def __init__(self, initial_consequences, trainable, with_norm):
        super().__init__()
        self.with_norm = with_norm
        self.size_out, self.size_in = initial_consequences.shape
        self.Z = nn.Parameter(initial_consequences.reshape(1, self.size_out, self.size_in), requires_grad=trainable)

    @classmethod
    def from_dimensions(cls, size_in, size_out, trainable=True, with_norm=True):
        return cls(torch.rand(size_out, size_in), trainable, with_norm)

    @classmethod
    def from_array(cls, initial_array, trainable=True, with_norm=True):
        return cls(torch.FloatTensor(np.array(initial_array)), trainable, with_norm)

    def forward(self, input: Tensor) -> Tensor:
        consequences = self.Z.expand(input.shape[0], self.size_out, self.size_in)
        if self.with_norm:
            norm_inp = input / (input.sum(-1, keepdim=True) + 1e-8)
            return torch.squeeze(torch.bmm(consequences, norm_inp.reshape(input.shape[0], self.size_in, 1)), 2)
        return torch.squeeze(torch.bmm(consequences, input.reshape(input.shape[0], self.size_in, 1)), 2)
