# Explainable fuzzy predictive maintenance — minimal reference

Minimal code accompanying *Explainable Predictive Condition-based Maintenance of Naval Propulsion Systems using Fuzzy Logic*.

It contains a small residual neural-network teacher with a differentiable fuzzy classifier head, and a local fuzzy decision tree that produces IF–THEN rules for an individual prediction.

## Run

```bash
pip install -r requirements.txt
python example.py
```

The example generates a small synthetic propulsion-like dataset, trains the teacher, predicts one query point, selects a class-stratified local neighbourhood using the teacher's predicted classes, fits a fuzzy decision tree, and prints local rules plus input-gradient saliency. Operational variables to exclude from tree splits are supplied through the tree constructor.

## Contents

- `fuzzy_head.py` — Gaussian fuzzy membership and normalized rule-consequence layers.
- `model.py` — small residual MLP teacher using the fuzzy head.
- `fuzzy_tree.py` — Gaussian-membership fuzzy decision tree and rule extraction.
- `synthetic_data.py` — deterministic synthetic data used only for the example.
- `example.py` — the complete training and explanation pipeline.

The optional plotting helpers in `fuzzy_tree.py` require Matplotlib and Graphviz. They are not included in `requirements.txt`, because they are not needed for the example.

## Scope

This is a compact, runnable reference implementation, not the full experimental codebase and not a reproduction of the paper's reported performance. The default data are synthetic so the mechanism can be run without data download or preprocessing. For the study, the public Condition Based Maintenance of Naval Propulsion Plants dataset was used; its data description and download are available from the [University of Strathclyde dataset record](https://pureportal.strath.ac.uk/en/datasets/condition-based-maintenance-of-naval-propulsion-plants-data-set-v/).

The local rules explain the teacher's behaviour around the selected query point. They should not be read as global physical laws or causal conclusions.

## Citation

Please cite the accompanying paper when using this code.
D. Kalogeropoulos, G. Sovatzidi, and D. K. Iakovidis, “Explainable neuro-fuzzy prediction for trustworthy decision-making in maritime,” in Proc. 34th Eur. Signal Process. Conf. (EUSIPCO), Bruges, Belgium, 2026, pp. 2601–XXXX
