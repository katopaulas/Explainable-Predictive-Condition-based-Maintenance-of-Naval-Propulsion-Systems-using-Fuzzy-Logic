"""Run: python example.py"""

import numpy as np
import torch
from synthetic_data import make_data
from model import FuzzyTeacher
from fuzzy_tree import FuzzyDecisionTree


def stratified_local_neighbourhood(X, query, teacher_labels, k=180):
    """Local neighbourhood: nearest k/n_classes samples from each predicted class."""
    labels = np.unique(teacher_labels)
    per_class = max(1, k // len(labels))
    distances = np.linalg.norm(X - query, axis=1)
    selected = []
    for label in labels:
        candidates = np.flatnonzero(teacher_labels == label)
        selected.append(candidates[np.argsort(distances[candidates])[:per_class]])
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


def explain_test_sample(teacher, X_train, train_teacher_labels, query, feature_names,
                        banned_features, neighbourhood_size=180):
    """Fit and report one local FDT explanation for one test-sample query."""
    local_positions = stratified_local_neighbourhood(
        X_train, query, train_teacher_labels, k=neighbourhood_size
    )
    X_local = X_train[local_positions]
    y_local = train_teacher_labels[local_positions]
    tree = FuzzyDecisionTree(
        max_depth=2,
        feature_names=feature_names,
        banned_features=banned_features,
    )
    tree.fit(X_local, y_local)
    query_tensor = torch.tensor(query[None, :])
    with torch.no_grad():
        teacher_prediction = teacher(query_tensor).argmax(1).item()
    saliency = teacher.saliency_gradient(query_tensor, teacher_prediction)[0].numpy()
    return {
        "teacher_prediction": teacher_prediction,
        "tree_prediction": tree.predict(query[None, :])[0],
        "fidelity": (tree.predict(X_local) == y_local).mean(),
        "saliency": saliency,
        "rules": tree.extract_rules(feature_names, query=query),
        "neighbourhood_size": len(local_positions),
    }


def main():
    np.random.seed(7)
    torch.manual_seed(7)
    X, y, feature_names = make_data()
    order = np.random.permutation(len(X))
    train_idx, test_idx = order[:900], order[900:]
    mean, std = X[train_idx].mean(0), X[train_idx].std(0) + 1e-6
    X_scaled = (X - mean) / std
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]

    teacher = train_dummy_model(X_train, y[train_idx], X.shape[1])
    teacher.eval()
    with torch.no_grad():
        train_teacher_labels = teacher(torch.tensor(X_train)).argmax(1).numpy()
        test_predictions = teacher(torch.tensor(X_test)).argmax(1).numpy()
    print(f"Teacher test accuracy: {(test_predictions == y[test_idx]).mean():.3f}")

    banned = ("lever_position",)
    for position, (sample_index, query, true_label, teacher_label) in enumerate(
        zip(test_idx, X_test, y[test_idx], test_predictions), start=1
    ):
        report = explain_test_sample(
            teacher, X_train, train_teacher_labels, query, feature_names, banned
        )
        print(f"\nTest sample {position}/{len(test_idx)} (index={sample_index})")
        print(f"  true class={true_label}, teacher class={teacher_label}, "
              f"FDT class={report['tree_prediction']}, "
              f"local FDT fidelity={report['fidelity']:.3f}, "
              f"neighbours={report['neighbourhood_size']}")
        print("  Feature attributions (Eq. 3 |dY/dx|):")
        for i in np.argsort(-report["saliency"]):
            marker = "*" if feature_names[i] in banned else ""
            print(f"    {feature_names[i]}{marker}: {report['saliency'][i]:.4f}")
        print("  FDT rules (support, confidence, and query firing strength):")
        for rule in report["rules"]:
            print(f"    {rule}")


if __name__ == "__main__":
    main()
