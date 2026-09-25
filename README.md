# Explainable fuzzy predictive maintenance — minimal reference

Minimal research demo accompanying *Explainable Predictive Condition-based Maintenance of Naval Propulsion Systems using Fuzzy Logic*. This is intentionally small example code for researchers to read and adapt, not a production-ready package or the full experimental implementation.

It contains a small residual neural-network teacher with a differentiable fuzzy classifier head, and a local fuzzy decision tree that produces IF–THEN rules for an individual prediction.

## Run

```bash
pip install -r requirements.txt
python example.py
```

The example generates a small synthetic propulsion-like dataset, trains the teacher, predicts the test set, and then explains every test sample. For each sample it selects a class-stratified local neighbourhood using the teacher's predicted classes, fits a fuzzy decision tree, computes gradient feature attributions, and prints the local FDT rules with their support, confidence, and query firing strength. Operational variables to exclude from tree splits are supplied through the tree constructor (`banned_features`).

Each nontrivial rule reports its firing strength for the query. Support is the sum of path firing strengths across neighbours; confidence is the fraction of that support agreeing with the rule's predicted class. These reported statistics exclude the class-balancing weights used for fitting, so confidence can fall below `1/n_classes`. They describe agreement with the teacher in the selected neighbourhood, not physical reliability or population-wide precision. The strongest firing rule is highlighted, but prediction aggregates contributions from all leaves.

Teacher test accuracy is printed before the per-sample reports. The reported local FDT fidelity is measured on the same neighbourhood used to fit the tree.

## Contents

- `fuzzy_head.py` — exponential fuzzy membership, `exp(-||Ax + b||)`, and normalized rule-consequence layers. `FuzzyTeacher.firing_strength` exposes total rule activation as a diagnostic; it is not a validated out-of-distribution detector.
- `model.py` — small residual MLP teacher using the fuzzy head.
- `fuzzy_tree.py` — Gaussian-membership fuzzy decision tree and rule extraction.
- `synthetic_data.py` — deterministic synthetic data used only for the example.
- `example.py` — the complete training and explanation pipeline.

The optional plotting helpers in `fuzzy_tree.py` require Matplotlib and Graphviz. They are not included in `requirements.txt`, because they are not needed for the example.

## Scope

This is a compact, runnable reference implementation, not the full experimental codebase. The default data are synthetic so the mechanism can be run without data download or preprocessing. For the study, the public Condition Based Maintenance of Naval Propulsion Plants dataset was used; its data description and download are available from the [University of Strathclyde dataset record](https://pureportal.strath.ac.uk/en/datasets/condition-based-maintenance-of-naval-propulsion-plants-data-set-v/).

## Correspondence with the paper

Defaults follow the paper: Gaussian set centres are placed by linear interpolation between the minimum and maximum feature values of the local neighbourhood (Sec. 2.2), and rule support and confidence are the Eq. 5 path firing strengths over that neighbourhood, with every neighbour weighted equally.

`FuzzyTeacher.saliency_gradient` implements Eq. 3, the absolute gradient of the selected defuzzified model output with respect to the input features. In this pipeline, inputs are standardised, so components are measured per training-set standard deviation of each feature. Eq. 4 in the paper defines rule support; it is reported for each extracted FDT rule.

## Split thresholds and root-only trees

The implementation sets these attributes in `FuzzyDecisionTree.__init__`:

```python
self.min_gain = 5e-3
self.min_gain_ratio = 0.01
```

For parent entropy `H` and best candidate information gain `G`, a split must satisfy `G >= 0.005` and, when `H > 1e-12`, `G / H >= 0.01`. Entropy uses base-2 logarithms, so the absolute threshold is in bits. Despite its name, `min_gain_ratio` measures the fraction of parent entropy removed; it is not the C4.5 gain ratio based on split entropy. These are demo heuristics, not universal or paper-derived constants. They can be changed on the tree instance before `fit()`.

Larger thresholds suppress weak splits and may leave only the root. Smaller thresholds allow more splits but can fit noise; lowering them does not guarantee a useful explanation. A root-only tree can also result from constant or uninformative features, a single class, excluded features, depth or sample limits, or post-pruning. In particular, `min_support` can collapse a split when all its children are leaves and any child has insufficient fuzzy support. `min_confidence` is a diagnostic threshold, not a pruning guarantee.

A root-only `IF TRUE` rule is a valid fallback: it means no conditional explanation survived the chosen settings. Inspect `tree.get_pruning_report()` and the neighbourhood before changing thresholds. If `root_was_leaf` is true, construction stopped before creating a split; otherwise the report records the pruning collapses. The current demo's highlighting code expects a `fires=` field, which root-only rules do not include: when adapting to data that produces this case, print the fallback rule directly rather than parsing that field.

## Citation

Please cite the accompanying paper when using this code.
D. Kalogeropoulos, G. Sovatzidi, and D. K. Iakovidis, “Explainable neuro-fuzzy prediction for trustworthy decision-making in maritime,” in Proc. 34th Eur. Signal Process. Conf. (EUSIPCO), Bruges, Belgium, 2026, pp. 2601–2605.
