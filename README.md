# Explainable fuzzy predictive maintenance — minimal reference

Minimal research demo accompanying *Explainable Predictive Condition-based Maintenance of Naval Propulsion Systems using Fuzzy Logic*. This is intentionally small example code for researchers to read and adapt, not a production-ready package or the full experimental implementation.

It contains a small residual neural-network teacher with a differentiable fuzzy classifier head, and a local fuzzy decision tree that produces IF–THEN rules for an individual prediction.

## Run

```bash
pip install -r requirements.txt
python example.py
```

The example generates a small synthetic propulsion-like dataset, trains the teacher, predicts one query point, selects a class-stratified local neighbourhood using the teacher's predicted classes, fits a fuzzy decision tree, and prints local rules plus input-gradient saliency. No distance kernel is applied by default. Operational variables to exclude from tree splits are supplied through the tree constructor (`banned_features`).

Each nontrivial rule reports its firing strength for the query. Support is the sum of path firing strengths across neighbours; confidence is the fraction of that support agreeing with the rule's predicted class. These reported statistics exclude the class-balancing weights used for fitting, so confidence can fall below `1/n_classes`. They describe agreement with the teacher in the selected neighbourhood, not physical reliability or population-wide precision. The strongest firing rule is highlighted, but prediction aggregates contributions from all leaves.

Teacher test accuracy is printed alongside a logistic-regression baseline. The reported surrogate fidelity is **training fidelity** on the same neighbourhood used to fit the tree; it is not an independent evaluation. When adapting the demo, also check agreement at the query and on held-out nearby samples.

## Contents

- `fuzzy_head.py` — exponential (Laplace-type) fuzzy membership, `exp(-||Ax + b||)`, and normalized rule-consequence layers. `FuzzyTeacher.firing_strength` exposes total rule activation as a diagnostic; it is not a validated out-of-distribution detector or an automatic abstention policy.
- `model.py` — small residual MLP teacher using the fuzzy head.
- `fuzzy_tree.py` — Gaussian-membership fuzzy decision tree and rule extraction.
- `synthetic_data.py` — deterministic synthetic data used only for the example.
- `example.py` — the complete training and explanation pipeline.

The optional plotting helpers in `fuzzy_tree.py` require Matplotlib and Graphviz. They are not included in `requirements.txt`, because they are not needed for the example.

## Scope

This is a compact, runnable reference implementation, not the full experimental codebase and not a reproduction of the paper's reported performance. The default data are synthetic so the mechanism can be run without data download or preprocessing. For the study, the public Condition Based Maintenance of Naval Propulsion Plants dataset was used; its data description and download are available from the [University of Strathclyde dataset record](https://pureportal.strath.ac.uk/en/datasets/condition-based-maintenance-of-naval-propulsion-plants-data-set-v/).

The local rules explain the teacher's behaviour around the selected query point. They should not be read as global physical laws or causal conclusions.

## Correspondence with the paper

Defaults follow the paper: Gaussian set centres are placed by linear interpolation between the minimum and maximum feature values of the local neighbourhood (Sec. 2.2), and rule support and confidence are the Eq. 5 path firing strengths over that neighbourhood, with every neighbour weighted equally. Two optional extensions are off by default and documented where they are defined: `FuzzyDecisionTree(partition_quantile=q)` trims the partition range to the q / 1-q quantiles, and passing `locality_kernel(...)` as `sample_weight` to `fit` makes the surrogate distance-weighted. Either one changes the reported statistics away from Eq. 5.

Both saliency definitions are implemented separately on `FuzzyTeacher`: `saliency_probability` is Eq. 3, the gradient of the softmax probability (not of the Eq. 2 defuzzified output), and `saliency_rule_activation` is Eq. 4, the summed absolute gradients of the top-k rule activations. Both return gradients in the units of the model input, which in this pipeline is standardised, so components are per standard deviation of each feature.

Implementation choices not fully specified by the paper include class-stratified neighbourhood selection, class-balanced fitting, and the prediction aggregation convention. Memberships are normalised across each feature's sets for fitting, inference, and rule statistics. Up to numerical tolerances, this conserves membership mass and gives nonnegative fuzzy information gain. Prediction mixes the class-balanced class distributions stored in the leaves using path firing strengths. These scores are not necessarily calibrated probabilities and differ from the empirical confidence printed for a rule. This is a documented interpretation, not a uniquely specified paper algorithm.

## Split thresholds and root-only trees

The implementation sets these attributes in `FuzzyDecisionTree.__init__`:

```python
self.min_gain = 5e-3
self.min_gain_ratio = 0.01
```

For parent entropy `H` and best candidate information gain `G`, a split must satisfy `G >= 0.005` and, when `H > 1e-12`, `G / H >= 0.01`. Entropy uses base-2 logarithms, so the absolute threshold is in bits. Despite its name, `min_gain_ratio` measures the fraction of parent entropy removed; it is not the C4.5 gain ratio based on split entropy. These are demo heuristics, not universal or paper-derived constants. They can be changed on the tree instance before `fit()`.

Larger thresholds suppress weak splits and may leave only the root. Smaller thresholds allow more splits but can fit noise; lowering them does not guarantee a useful explanation. A root-only tree can also result from constant or uninformative features, a single class, excluded features, depth or sample limits, or post-pruning. In particular, `min_support` can collapse a split when all its children are leaves and any child has insufficient fuzzy support. `min_confidence` is a diagnostic threshold, not a pruning guarantee.

A root-only `IF TRUE` rule is a valid fallback: it means no conditional explanation survived the chosen settings. Inspect `tree.get_pruning_report()` and the neighbourhood before changing thresholds. If `root_was_leaf` is true, construction stopped before creating a split; otherwise the report records the pruning collapses. The current demo's highlighting code expects a `fires=` field, which root-only rules do not include: when adapting to data that produces this case, print the fallback rule directly rather than parsing that field.

## Adapting the demo

Use finite numeric arrays, aligned labels, and train-only preprocessing. The small implementation does not comprehensively validate inputs or hyperparameters. If using optional sample weights, supply finite, nonnegative weights with positive total mass; use `0 <= partition_quantile < 0.5`. Predictions far outside the fitted neighbourhood can lose all Gaussian activation and return zero scores; check coverage and handle this case before interpreting an argmax. Neural rule normalization also becomes inaccurate when total activation is extremely small.

Call `teacher.eval()` for saliency. For batched probability saliency, omit `target` or supply one target index per sample; the scalar-target path is intended for the single query shown in the demo. The alternate `from_centers_and_scales` initializer does not currently preserve requested centres for non-unit scales; the demo uses `from_dimensions` instead. Optional membership plotting expects `outputs/plots/memberships/` to exist.

The documented synthetic example runs end to end. Adapting it to another dataset requires choosing and validating neighbourhood size, exclusions, tree thresholds, and evaluation protocol. Dependency versions are not pinned, and the repository does not currently include an automated regression suite. This demo is a starting point for that work, not evidence of deployment readiness or reproduction of the study's results.

## Citation

Please cite the accompanying paper when using this code.
