"""A small propulsion-like binary degradation dataset for the runnable example."""

import numpy as np

FEATURE_NAMES = ("lever_position", "GGn", "T2", "P2", "fuel_flow")


def make_data(n_samples=1200, seed=7):
    rng = np.random.default_rng(seed)
    lever = rng.integers(1, 10, n_samples)
    degradation = rng.uniform(0, 1, n_samples)
    noise = lambda scale: rng.normal(0, scale, n_samples)

    # Sensor responses combine operating load and a latent compressor degradation state.
    ggn = 6500 + 95 * lever + 220 * degradation + noise(35)
    t2 = 500 + 9 * lever + 48 * degradation + noise(4)
    p2 = 12 + 0.35 * lever - 1.1 * degradation + noise(0.18)
    fuel_flow = 0.15 + 0.035 * lever + 0.01 * degradation + noise(0.008)
    X = np.column_stack((lever, ggn, t2, p2, fuel_flow)).astype(np.float32)
    y = (degradation >= 0.55).astype(np.int64)
    return X, y, FEATURE_NAMES
