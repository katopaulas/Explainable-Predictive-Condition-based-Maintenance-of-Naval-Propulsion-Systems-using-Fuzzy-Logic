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


def locality_kernel(X, query):
    """Optional extension, unused by default.

    The paper weights every neighbour equally (Sec. 2.2, Eq. 5), so the default
    pipeline passes no sample_weight. Supplying this kernel to
    FuzzyDecisionTree.fit turns the surrogate into a LIME-style distance-weighted
    fit; support and confidence then become kernel-weighted rather than Eq. 5
    quantities. Normalised to mean 1 so min_support keeps its sample-count
    meaning.
    """
    d = np.linalg.norm(X - query, axis=1)
    width = np.median(d) + 1e-12
    w = np.exp(-(d / width) ** 2)
    return w / w.mean()


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


def logistic_baseline(X_train, y_train, X_test, y_test, epochs=400):
    """Linear reference point for the teacher's accuracy."""
    model = torch.nn.Linear(X_train.shape[1], 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    x, y = torch.tensor(X_train), torch.tensor(y_train)
    for _ in range(epochs):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step()
    with torch.no_grad():
        return (model(torch.tensor(X_test)).argmax(1).numpy() == y_test).mean()


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
    baseline = logistic_baseline(X_scaled[train_idx], y[train_idx],
                                 X_scaled[test_idx], y[test_idx])
    print(f"Teacher test accuracy: {accuracy:.3f}  "
          f"(logistic-regression baseline {baseline:.3f}; this synthetic task is "
          f"linear by construction, so the baseline is expected to be competitive)")

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

    banned = ("lever_position",)
    tree = FuzzyDecisionTree(
        max_depth=2,
        feature_names=feature_names,
        banned_features=banned,
    )
    tree.fit(X_scaled[local_idx], local_teacher_labels)
    fidelity = (tree.predict(X_scaled[local_idx]) == local_teacher_labels).mean()
    print(f"Query prediction: {'degraded' if query_prediction else 'healthy'}")
    print(f"Surrogate fidelity to the teacher on the neighbourhood: {fidelity:.3f}")
    print("Local fuzzy rules (support and confidence are Eq. 5 path firing strengths "
          "over the neighbourhood; 'fires' = strength for this query):")
    rules = tree.extract_rules(feature_names, query=X_scaled[query_index])
    firing = [float(r.split("fires=")[1].rstrip("]")) for r in rules]
    for rule, f in zip(rules, firing):
        print(f"  {'->' if f == max(firing) else '  '} {rule}")

    query = torch.tensor(X_scaled[query_index:query_index + 1])
    eq3 = teacher.saliency_probability(query, query_prediction)[0].numpy()
    eq4 = teacher.saliency_rule_activation(query, k=3)[0].numpy()
    print("Saliency (gradients w.r.t. standardised inputs, i.e. per standard deviation "
          "of each feature; * = excluded from tree splits):")
    print(f"   {'feature':<16}{'Eq.3 |dp/dx|':>14}{'Eq.4 top-3 rules':>19}")
    for i in np.argsort(-eq3):
        star = "*" if feature_names[i] in banned else " "
        print(f"   {feature_names[i] + star:<16}{eq3[i]:>14.4f}{eq4[i]:>19.4f}")


if __name__ == "__main__":
    main()
