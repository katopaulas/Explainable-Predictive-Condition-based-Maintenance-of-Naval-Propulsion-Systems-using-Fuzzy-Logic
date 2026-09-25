"""Regression tests for consequent voting and faithful merged-rule reports."""
import re
import unittest

import numpy as np

from fuzzy_tree import FuzzyDecisionTree, FuzzyNode


# Rule strengths are printed to six decimals, so scores reconstructed from the
# printed text agree with predict_proba() to ~1e-6, not to machine precision.
PRINTED = dict(atol=1e-6)


def printed_scores(tree, query):
    scores = {c: 0.0 for c in tree.classes_}
    for rule in tree.extract_rules(query=query):
        label = int(re.search(r"THEN class = (\d+)", rule)[1])
        fire = float(re.search(r"fires=([^\]]+)", rule)[1])
        scores[label] += fire
    total = sum(scores.values())
    return np.array([scores[c] / total for c in tree.classes_])


class RuleReportingTests(unittest.TestCase):
    def test_impure_same_class_leaves_merge_without_changing_votes(self):
        tree = FuzzyDecisionTree(feature_names=['x'])
        tree.classes_ = np.array([0, 1])
        leaves = []
        for label, support, confidence in [(1, 90, .6), (1, 10, .9), (0, 20, .8)]:
            masses = {label: support * confidence, 1-label: support * (1-confidence)}
            leaves.append(FuzzyNode(label=label, support=support, confidence=confidence,
                                    class_masses=masses, raw_class_masses=masses))
        tree.root = FuzzyNode(feature=0, centers=[-1, 0, 1], sigmas=[1, 1, 1],
                              children=list(enumerate(leaves)))
        query = np.array([0.0])
        memberships = np.exp(-.5 * np.array([1., 0., 1.]))
        expected = np.array([memberships[2], memberships[0]+memberships[1]])
        expected /= expected.sum()
        np.testing.assert_allclose(list(tree.predict_proba([query])[0].values()), expected)
        np.testing.assert_allclose(printed_scores(tree, query), expected, **PRINTED)
        self.assertIn('confidence=0.63', tree.extract_rules(query=query)[0])

    def test_fitted_trees_reconstruct_and_track_only_surviving_features(self):
        for seed in range(10):
            rng = np.random.default_rng(seed)
            X = rng.normal(size=(100, 3))
            y = (X[:, 0] + X[:, 1] > .6).astype(int)
            tree = FuzzyDecisionTree(max_depth=3, min_support=seed)
            tree.fit(X, y, sample_weight=rng.uniform(.2, 2, len(y)))
            used = set()
            def walk(node):
                if node.label is None:
                    used.add(node.feature)
                    for _, child in node.children:
                        walk(child)
            walk(tree.root)
            self.assertEqual(set(tree.fuzzy_params), used)
            for query in X[:10]:
                expected = list(tree.predict_proba([query])[0].values())
                np.testing.assert_allclose(printed_scores(tree, query), expected, **PRINTED)
            tree.max_depth = 0
            tree.fit(X, y)
            self.assertEqual(tree.fuzzy_params, {})
            np.testing.assert_allclose(printed_scores(tree, X[0]),
                                       list(tree.predict_proba(X[:1])[0].values()), **PRINTED)

    def test_no_activation_is_not_an_arbitrary_class(self):
        X = np.tile([[-1.], [1.]], (50, 1))
        tree = FuzzyDecisionTree(min_support=0)
        tree.fit(X, np.tile([0, 1], 50))
        with self.assertRaises(ValueError):
            tree.predict([[1e6]])


if __name__ == '__main__':
    unittest.main()
