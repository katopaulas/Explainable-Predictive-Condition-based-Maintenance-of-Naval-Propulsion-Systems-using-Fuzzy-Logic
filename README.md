# Explainable fuzzy predictive maintenance — minimal reference

Minimal research demo accompanying *Explainable Neuro-Fuzzy Prediction for Trustworthy Decision-Making in Maritime*. It implements the paper's core building blocks: a small residual neural network with a differentiable fuzzy classifier head and a local fuzzy decision tree that produces IF–THEN rules for individual predictions. It is intended for research and adaptation, not as a production-ready package or the full experimental implementation.

## Run

```bash
pip install -r requirements.txt
python example.py
```

The example generates a small synthetic propulsion-like dataset, trains the teacher, predicts the test set, and then explains every test sample. For each sample it selects a class-stratified local neighbourhood using the teacher's predicted classes, fits a fuzzy decision tree, computes gradient feature attributions, and prints the local FDT rules with their support, confidence, and query firing strength. Operational variables to exclude from tree splits are supplied through the tree constructor (`banned_features`).

Each nontrivial rule reports its firing strength for the query. Support is the sum of path firing strengths across neighbours; confidence is the fraction of that support agreeing with the rule's predicted class. These reported statistics exclude the class-balancing weights used for fitting, so confidence can fall below `1/n_classes`. They describe agreement with the teacher in the selected neighbourhood, not physical reliability or population-wide precision. Prediction aggregates contributions from all leaves.

Teacher test accuracy is printed before the per-sample reports. Neighbourhood training fidelity measures agreement on the samples used to fit the tree. Query agreement is reported separately for each sample and across the test set; disagreements are explicitly flagged.

Each leaf contributes its entire path firing strength only to its stated class. The class with the largest summed strength wins; confidence does not weight this vote. Same-class linguistic rules can be merged by adding their firing strengths without changing the prediction. `predict_proba()` returns normalized firing scores, not calibrated probabilities. Printed strengths retain enough precision to reconstruct those scores up to floating-point rounding. Confidence remains an empirical neighbourhood statistic, even when class balancing changes the class chosen at a leaf.

Feature attributions are printed in descending gradient magnitude with four decimal places.

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

The implementation follows the explainability workflow presented in *Explainable Neuro-Fuzzy Prediction for Trustworthy Decision-Making in Maritime*:

- The teacher combines a residual neural feature extractor with a differentiable fuzzy classifier head. Its membership function is $\mu_{f_i}(\mathbf{u}) = \exp(-\|A_{f_i}\mathbf{u}+b_{f_i}\|_2)$, and its defuzzification layer uses normalized rule activations.
- For each test sample, the teacher produces a prediction and a class-stratified k-nearest local neighbourhood is selected from the training set.
- A shallow FDT is fitted to the teacher predictions in that local neighbourhood. Gaussian fuzzy-set centres are initialized by linear interpolation between the local minimum and maximum of each feature. Splits use weighted fuzzy entropy and the tree is pruned after fitting.
- Feature attribution uses the standard-gradient saliency map in Eq. 3, $E_{\mathrm{grad}}(\mathbf{x}) = |\partial Y / \partial \mathbf{x}|$. The demo reports gradients of the selected defuzzified teacher output with respect to standardized input features.
- Each root-to-leaf path is reported as a fuzzy IF–THEN rule. Rule support and confidence follow Eqs. 4–5: $s(R_i)=\sum_{\mathbf{x}\in D}\alpha_{c_i}(\mathbf{x})$ and $p(R_i)=\sum_{\mathbf{x}\in D}\alpha_{c_i}(\mathbf{x}\mid\hat y=K) / s(R_i)$.

The code is a compact reference implementation. Tree depth, split thresholds, class balancing, neighbourhood size, and the aggregation of leaf activations are implementation settings that can be adapted for a different dataset or evaluation protocol.

## Split thresholds and root-only trees

The implementation sets these attributes in `FuzzyDecisionTree.__init__`:

```python
self.min_gain = 5e-3
self.min_gain_ratio = 0.01
```

For parent entropy `H` and best candidate information gain `G`, a split must satisfy `G >= 0.005` and, when `H > 1e-12`, `G / H >= 0.01`. Entropy uses base-2 logarithms, so the absolute threshold is in bits. Despite its name, `min_gain_ratio` measures the fraction of parent entropy removed; it is not the C4.5 gain ratio based on split entropy. These are demo heuristics, not universal or paper-derived constants. They can be changed on the tree instance before `fit()`.

Larger thresholds suppress weak splits and may leave only the root. Smaller thresholds allow more splits but can fit noise; lowering them does not guarantee a useful explanation. A root-only tree can also result from constant or uninformative features, a single class, excluded features, depth or sample limits, or post-pruning. In particular, `min_support` can collapse a split when all its children are leaves and any child has insufficient fuzzy support. `min_confidence` is a diagnostic threshold, not a pruning guarantee.

`min_samples` compares class-balanced fuzzy mass, rather than a count of distinct samples. Each feature can be used only once along a path. Partitions use the full selected neighbourhood at every depth, giving each linguistic term a consistent meaning across branches. Neighbourhood size and class stratification affect how local an explanation is and should be chosen for the application. Membership plots refer only to surviving split features.

A root-only `IF TRUE` rule is a valid fallback: it means no conditional explanation survived the chosen settings. Inspect `tree.get_pruning_report()` and the neighbourhood before changing thresholds. If `root_was_leaf` is true, construction stopped before creating a split; otherwise the report records the pruning collapses.

## Future extensions

The hooks below already exist in the FDT; future work is to evaluate them in the explanation pipeline. Both are unused by the default demo.

- Distance-weighted local surrogates: use a distance kernel through `FuzzyDecisionTree.fit(..., sample_weight=...)` to emphasize neighbours closest to the explained sample.
- Robust fuzzy partitions: initialize fuzzy-set ranges from quantiles rather than local minima and maxima to reduce sensitivity to extreme observations.

## Citation

Please cite the accompanying paper when using this code.

D. Kalogeropoulos, G. Sovatzidi, and D. K. Iakovidis, “Explainable neuro-fuzzy prediction for trustworthy decision-making in maritime,” in Proc. 34th Eur. Signal Process. Conf. (EUSIPCO), Bruges, Belgium, 2026, pp. 2601–2605.
