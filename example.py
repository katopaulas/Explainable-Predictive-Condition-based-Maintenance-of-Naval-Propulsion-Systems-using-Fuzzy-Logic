"""Run: python example.py"""

import numpy as np
import torch
from synthetic_data import make_data
from model import FuzzyTeacher
from fuzzy_tree import FuzzyDecisionTree


def stratified_local_neighbourhood(X, query, teacher_labels, k=180):
    """Local neighbourhood: nearest k/2 samples from each teacher-predicted class."""
    per_class = k // 2
    selected = []
    distances = np.linalg.norm(X - query, axis=1)
    for label in (0, 1):
        candidates = np.flatnonzero(teacher_labels == label)
        nearest = candidates[np.argsort(distances[candidates])[:min(per_class, len(candidates))]]
        selected.append(nearest)
    return np.concatenate(selected)


def train_dummy_model(X_train, y_train, input_size, epochs=150):
    """Train the small reference teacher used by this minimal example."""
    teacher = FuzzyTeacher(input_size)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=3e-3)
    x_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(teacher(x_train), y_train)
        loss.backward()
        optimizer.step()
    return teacher


def main():
    np.random.seed(7)
    torch.manual_seed(7)
    X, y, feature_names = make_data()
    order = np.random.permutation(len(X))
    train_idx, test_idx = order[:900], order[900:]
    mean, std = X[train_idx].mean(0), X[train_idx].std(0) + 1e-6
    X_scaled = (X - mean) / std

    teacher = train_dummy_model(X_scaled[train_idx], y[train_idx], X.shape[1])

    teacher.eval()
    with torch.no_grad():
        test_logits = teacher(torch.tensor(X_scaled[test_idx]))
        accuracy = (test_logits.argmax(1).numpy() == y[test_idx]).mean()
    print(f"Teacher test accuracy: {accuracy:.3f}")

    # Explain one test point: use a class-stratified local neighbourhood, then fit
    # an FDT to the teacher's decisions within that neighbourhood.
    query_index = test_idx[0]
    with torch.no_grad():
        train_teacher_labels = teacher(torch.tensor(X_scaled[train_idx])).argmax(1).numpy()
        query_prediction = teacher(torch.tensor(X_scaled[query_index:query_index + 1])).argmax(1).item()
    local_neighbourhood_positions = stratified_local_neighbourhood(
        X_scaled[train_idx], X_scaled[query_index], train_teacher_labels, k=180
    )
    local_idx = train_idx[local_neighbourhood_positions]
    local_teacher_labels = train_teacher_labels[local_neighbourhood_positions]

    tree = FuzzyDecisionTree(
        max_depth=2,
        feature_names=feature_names,
        banned_features=("lever_position",),
    )
    tree.fit(X_scaled[local_idx], local_teacher_labels)
    print(f"Query prediction: {'degraded' if query_prediction else 'healthy'}")
    print("Local fuzzy rules:")
    for rule in tree.extract_rules(feature_names):
        print(" ", rule)

    query = torch.tensor(X_scaled[query_index:query_index + 1], requires_grad=True)
    teacher(query)[0, query_prediction].backward()
    saliency = np.abs(query.grad.numpy()[0])
    print("Gradient saliency:", dict(zip(feature_names, np.round(saliency, 3))))


if __name__ == "__main__":
    main()
