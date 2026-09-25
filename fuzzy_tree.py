import numpy as np

def count_nodes(node):
    if node is None:
        return 0
    if node.label is not None:  # leaf
        return 1
    return 1 + sum(count_nodes(child) for _, child in node.children)

# -----------------------------
# Fuzzy membership
# -----------------------------
def gaussian(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2)

def weighted_majority(y, w):
    classes = np.unique(y)
    scores = {c: w[y == c].sum() for c in classes}
    return max(scores, key=scores.get)

def weighted_entropy(y, w):
    """Compute weighted entropy for fuzzy information gain."""
    classes = np.unique(y)
    total = w.sum()
    if total < 1e-12:
        return 0.0
    entropy = 0.0
    for c in classes:
        p = w[y == c].sum() / total
        if p > 1e-12:
            entropy -= p * np.log2(p)
    return entropy

# -----------------------------
# Fuzzy Tree Node
# -----------------------------
class FuzzyNode:
    '''
        support     : fuzzy sample count reaching this node (unweighted)
        confidence  : empirical fuzzy proportion of `label` among that support
        centers     : Fuzzy set centers for this feature split
        sigmas      : Fuzzy set spreads for this feature split
        e.g   RULE A     IF ...  THEN class = 2
                                 [support=23.7, confidence=0.91]

        class_masses drive fitting (balanced prior); raw_class_masses drive the
        reported statistics. `label` follows class_masses, so with a rare class
        confidence can legitimately fall below 1/n_classes.
    '''

    def __init__(self, feature=None, children=None, label=None,
                 support=None, confidence=None, centers=None, sigmas=None,
                 class_masses=None, raw_class_masses=None, balanced_confidence=None):
        self.feature = feature
        self.children = children  # [(label, node), ...] or None if leaf
        self.label = label
        self.support = support
        self.confidence = confidence
        self.centers = centers   # centers of fuzzy sets corresponding to children
        self.sigmas  = sigmas     
        self.class_masses = class_masses
        self.raw_class_masses = raw_class_masses
        self.balanced_confidence = balanced_confidence
        


# -----------------------------
# Fuzzy Decision Tree
# -----------------------------
class FuzzyDecisionTree:
    """
        n_partitions = number of fuzzy sets per feature (e.g., 3 → Low/Medium/High)
    """
    def __init__(self, max_depth=3, min_samples=10, n_partitions=3,
                 min_support=10, min_confidence=0.5, feature_names=None,
                 verbose=False, banned_features=None,
                 partition_quantile=0.0):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_partitions = n_partitions
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.feature_names = list(feature_names) if feature_names is not None else []
        self.root = None
        self.classes_ = None
        self.n_features_in_ = None
        self.verbose = verbose

        self.min_gain=5e-3; self.min_gain_ratio=0.01
        self.banned_features = set(banned_features) if banned_features is not None else set()
        self.partition_quantile = partition_quantile
        unknown = self.banned_features - set(self.feature_names)
        if unknown:
            raise ValueError(f"banned_features not in feature_names: {sorted(unknown)}")

    def _fname(self, f, feature_names=None):
        """Feature name: explicit argument, then constructor list, then fallback."""
        if f is None:
            return None
        if feature_names is not None and f < len(feature_names):
            return str(feature_names[f])
        if f < len(self.feature_names):
            return str(self.feature_names[f])
        return f"Feature_{f}"

    def _n_features(self):
        """Feature count from the fitted data; names are optional metadata."""
        if self.n_features_in_ is not None:
            return self.n_features_in_
        return len(self.feature_names)

    def _findex(self, name):
        known = [self._fname(f) for f in range(self._n_features())]
        if name in known:
            return known.index(name)
        raise ValueError(f"unknown feature {name!r}; known: {known}")

    def _set_label(self, k):
        return self.get_linguistic_labels(self.n_partitions)[k]

    def get_fuzzy_sets_adaptive_quantiles(self, X_column, feature_idx, feature_name='', plot=False):
        n_sets = self.n_partitions
        x = np.asarray(X_column)

        if n_sets == 1:
            c = np.median(x)
            s = np.std(x) + 1e-6
            return [c], np.array([s])

        # centers at quantiles (more where data are)
        qs = np.linspace(0, 1, n_sets)
        centers = np.quantile(x, qs)

        # sigma per center from neighbor spacing
        # (avoid zero spacing when quantiles repeat due to ties)
        centers = np.unique(centers)
        if len(centers) < n_sets:
            # fallback: spread evenly if too many ties
            c_min, c_max = x.min(), x.max()
            centers = np.linspace(c_min, c_max, n_sets)

        centers = np.asarray(centers)
        deltas = np.diff(centers)
        deltas = np.maximum(deltas, 1e-6)

        sigmas = np.empty_like(centers)
        sigmas[0]  = deltas[0]
        sigmas[-1] = deltas[-1]
        sigmas[1:-1] = 0.5 * (deltas[:-1] + deltas[1:])

        # overlap factor
        beta = 0.8
        sigmas = beta * sigmas + 1e-6

        return centers.tolist(), sigmas


    def get_fuzzy_sets(self, X_column, feature_idx, feature_name='', plot=False):
        """Gaussian set centres by linear interpolation between the minimum and
        maximum feature values in the local neighbourhood (paper, Sec. 2.2).

        Optional extension, off by default: partition_quantile > 0 interpolates
        between the q and 1-q quantiles instead. min/max have breakdown point
        1/n, so one spiked sensor reading drags the whole partition with it and
        the Low/Medium/High labels stop meaning anything; trimming bounds that
        at the cost of departing from the published method.
        """
        n_sets = self.n_partitions
        q = self.partition_quantile
        if q > 0:
            c_min, c_max = np.quantile(X_column, [q, 1 - q])
            if c_max - c_min < 1e-12:            # degenerate after trimming
                c_min, c_max = X_column.min(), X_column.max()
        else:
            c_min, c_max = X_column.min(), X_column.max()
        if n_sets == 1:
            centers = [(c_min + c_max)/2]
            sigmas = [(c_max - c_min)/2 + 1e-6]
        else:
            centers = np.linspace(c_min, c_max, n_sets)
            sigma = (c_max - c_min) / (2 * (n_sets - 1)) + 1e-6
            sigmas = [sigma] * n_sets

        return centers, np.array(sigmas)
        

    def fit(self, X, y, sample_weight=None):
        """`sample_weight` is the locality kernel for a local surrogate: it scales
        both the fitting weights and the reported support/confidence, so the
        statistics describe the neighbourhood rather than the raw sample."""
        self.n_features_in_ = X.shape[1]
        self.classes_, counts = np.unique(y, return_counts=True)
        # Weight = Total Samples / (Number of Classes * Class Count)
        class_weights = {class_id: len(y) / (len(self.classes_) * count) for class_id, count in zip(self.classes_, counts)}
        w = np.array([class_weights[class_id] for class_id in y])
        w_raw = np.ones(len(y), dtype=float)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float)
            if sw.shape != (len(y),):
                raise ValueError(f"sample_weight must have shape {(len(y),)}, got {sw.shape}")
            w, w_raw = w * sw, w_raw * sw
        self.root = self._build(X, y, w, w_raw, depth=0)
        n_nodes = count_nodes(self.root)
        

        # Automatically prune after building
        self.prune(min_support=self.min_support, min_confidence=self.min_confidence)
        n_nodes2 = count_nodes(self.root)

        if self.verbose:
            print(f'Trained tree with {n_nodes} nodes')
            if n_nodes != n_nodes2:
                print(f'From {n_nodes} nodes, pruned to {n_nodes2} nodes')
            print()


    def _build(self, X, y, w, w_raw, depth, features_used=None):
        n_features = X.shape[1]
        class_masses = {label: float(w[y == label].sum()) for label in self.classes_}
        raw_class_masses = {label: float(w_raw[y == label].sum()) for label in self.classes_}
        if features_used is None:
            features_used=[]
        #print(features_used)
        # Stop condition
        stop_condition = depth >= self.max_depth or w.sum() < self.min_samples or len(features_used) >= n_features
        if not stop_condition:
            best_f, best_score = None, -np.inf
            best_centers, best_sigmas = None, None
            
            # Calculate parent entropy for information gain
            parent_entropy = weighted_entropy(y, w)

            for f in range(n_features):
                if f in features_used or self._fname(f) in self.banned_features: # avoid immediate reuse
                    continue
                f_name = self._fname(f)
                x_local_col = X[:, f]
                centers, sigmas = self.get_fuzzy_sets(x_local_col, f, feature_name=f_name, plot=False)
                #mu_sets = [gaussian(X[:, f], c, s) for c, s in zip(centers, sigmas)]
                mu_sets = np.stack([gaussian(X[:, f], c, s) for c, s in zip(centers, sigmas)])
                mu_sets = mu_sets / (mu_sets.sum(axis=0, keepdims=True) + 1e-12)

                # Calculate weighted entropy after split (information gain approach)
                total_weight = w.sum()
                weighted_child_entropy = 0.0
                for mu in mu_sets:
                    w_child = w * mu
                    weight_ratio = w_child.sum() / total_weight
                    if weight_ratio > 1e-6:
                        child_entropy = weighted_entropy(y, w_child)
                        weighted_child_entropy += weight_ratio * child_entropy
                # Information gain (higher is better)
                score = parent_entropy - weighted_child_entropy
                # if score < 0:
                #     print()

                # # Weighted variance over all sets approach
                # score = np.sum([np.average((mu - mu.mean())**2, weights=w) for mu in mu_sets])
                if score > best_score:
                    best_score = score
                    best_f = f
                    best_centers, best_sigmas = centers, sigmas
                    self.fuzzy_params = getattr(self, "fuzzy_params", {})
                    self.fuzzy_params[best_f] = (best_centers, best_sigmas)

            if best_f is None or best_score < self.min_gain or (parent_entropy > 1e-12 and (best_score / parent_entropy) < self.min_gain_ratio):
                stop_condition = True

        if not stop_condition:
            # Compute memberships for the chosen feature
            mu_sets = np.stack([gaussian(X[:, best_f], c, s)
                                for c, s in zip(best_centers, best_sigmas)
                                ])  # shape (K, N)
            mu_sets = mu_sets / (mu_sets.sum(axis=0, keepdims=True) + 1e-12)
            #winner = np.argmax(mu_sets, axis=0)  # dominant fuzzy set per sample
               
            
            # Prepare children nodes, keyed by fuzzy-set index
            children_nodes = []
            for i in range(self.n_partitions):
                #mask = (winner == i)
                #if not np.any(mask):
                #    continue
                #w_child = w * mu_sets[i]*mask

                w_child = w * mu_sets[i]
                if w_child.sum() > 1e-6:
                    new_features_used = features_used + [best_f]
                    child_node = self._build(X, y, w_child, w_raw * mu_sets[i],
                                             depth + 1, new_features_used)
                    children_nodes.append((i, child_node))

            # Degenerate case: if no children, make leaf
            if len(children_nodes) == 0:
                stop_condition = True


        # ---------- LEAF ----------(?)
        if stop_condition:
            label = weighted_majority(y, w)
            support = float(w_raw.sum())
            return FuzzyNode(
                label=label, support=support,
                confidence=raw_class_masses[label] / (support + 1e-12),
                balanced_confidence=class_masses[label] / (w.sum() + 1e-12),
                class_masses=class_masses, raw_class_masses=raw_class_masses,
            )

        # ---------- INTERNAL NODE ----------
        return FuzzyNode(
            feature=best_f,
            children=children_nodes,
            centers=best_centers,
            sigmas=best_sigmas,
            support=float(w_raw.sum()),
            class_masses=class_masses,
            raw_class_masses=raw_class_masses,
        )
    

    def prune(self, min_support=1.0, min_confidence=0.5):
        """
        Bottom-up, coverage-preserving post-pruning.

        A low-support fuzzy child cannot simply be removed: that would leave
        part of the fuzzy partition without a prediction. If a direct terminal
        child fails ``min_support``, its entire split is collapsed to the
        parent-majority leaf. Replacement statistics are calculated from exact
        fuzzy class masses saved at fitting time.

        ``min_confidence`` is recorded as a rule-quality diagnostic, rather
        than used for structural collapse. Local neighbourhoods are selected
        class-balanced, so using it to collapse branches can turn otherwise
        useful explanations into a root-only 50/50 fallback.
        """
        report = {
            "min_support": min_support,
            "min_confidence": min_confidence,
            "nodes_before": count_nodes(self.root),
            "collapsed_splits": [],
            "low_confidence_terminal_children": [],
        }

        _feature_name = self._fname

        def _collapse_to_leaf(node, path, reason, weak_children=None):
            masses = node.class_masses
            if not masses:
                raise RuntimeError("Cannot prune a node without class masses")
            split_feature = _feature_name(node.feature)
            labels_before = [child.label for _, child in node.children] if node.children else []
            raw = node.raw_class_masses
            node.label = max(masses, key=masses.get)
            node.support = sum(raw.values())
            node.confidence = raw[node.label] / (node.support + 1e-12)
            node.balanced_confidence = masses[node.label] / (sum(masses.values()) + 1e-12)
            node.feature = None
            node.children = None
            node.centers = None
            node.sigmas = None
            report["collapsed_splits"].append({
                "path": tuple(path),
                "feature": split_feature,
                "reason": reason,
                "labels_before": tuple(labels_before),
                "weak_children": weak_children or [],
                "replacement_label": node.label,
                "replacement_support": node.support,
                "replacement_confidence": node.confidence,
            })
            return node

        def _has_low_support(node):
            return node.support < min_support

        def _prune_node(node, path=()):
            if node.label is not None:
                return node

            node.children = [
                (k, _prune_node(child, path + (f"{_feature_name(node.feature)}:{self._set_label(k)}",)))
                for k, child in node.children
            ]

            all_children_are_leaves = all(child.label is not None for _, child in node.children)
            if not all_children_are_leaves:
                return node

            labels = [child.label for _, child in node.children]  
            low_support_children = []
            for k, child in node.children:
                child_report = {
                    "path": tuple(path + (f"{_feature_name(node.feature)}:{self._set_label(k)}",)),
                    "set": self._set_label(k),
                    "label": child.label,
                    "support": child.support,
                    "confidence": child.confidence,
                }
                if child.confidence < min_confidence:
                    report["low_confidence_terminal_children"].append(child_report)
                if _has_low_support(child):
                    low_support_children.append(child_report)
            if len(set(labels)) == 1:
                return _collapse_to_leaf(node, path, "redundant_split")
            if low_support_children:
                return _collapse_to_leaf(
                    node, path, "low_support_terminal_child", low_support_children
                )

            return node
        
        if self.root:
            root_was_leaf = self.root.label is not None
            self.root = _prune_node(self.root)
            report["nodes_after"] = count_nodes(self.root)
            report["nodes_pruned"] = report["nodes_before"] - report["nodes_after"]
            report["root_only"] = self.root.label is not None
            report["root_was_leaf"] = root_was_leaf
            self.pruning_report_ = report

            if self.verbose:
                print(
                    "Pruning: "
                    f"{report['nodes_before']} -> {report['nodes_after']} nodes; "
                    f"collapsed {len(report['collapsed_splits'])} split(s)."
                )
                if report["low_confidence_terminal_children"]:
                    print(
                        "Pruning diagnostic: "
                        f"{len(report['low_confidence_terminal_children'])} low-confidence "
                        "terminal child(ren) observed."
                    )
                if report["root_only"]:
                    print("Pruning outcome: root-only fallback rule.")

    def get_pruning_report(self):
        """Return diagnostics from the latest fit/prune operation."""
        return getattr(self, "pruning_report_", None)


    def _predict_sample(self, x, node, strength=1.0):
        """
        Recursively compute class vote for a single sample x.
        strength = current membership weight reaching this node
        """
        if node.label is not None:
            # Leaf: distribute the strength over the classes present in the leaf,
            # not onto the winner alone.
            total = sum(node.class_masses.values())
            if total < 1e-12:
                return {node.label: strength}
            return {cls: strength * m / total for cls, m in node.class_masses.items()}

        # Compute memberships for this feature
        mu = np.array([
            gaussian(x[node.feature], c, s)
            for c, s in zip(node.centers, node.sigmas)
        ])
        mu = mu / (mu.sum() + 1e-12)
        # Internal node: loop over all children
        votes = {}
        for k, child_node in node.children:
            child_strength = strength * mu[k]
            if child_strength > 1e-6:
                child_votes = self._predict_sample(x, child_node, child_strength)
                for cls, v in child_votes.items():
                    votes[cls] = votes.get(cls, 0) + v
        return votes
    

    def predict_proba(self, X):
        """Aggregate leaf activations into a class distribution.

        The paper states only that "final predictions are determined by
        aggregating the firing strength of the activated tree nodes", which does
        not uniquely determine the aggregation. Three readings are consistent
        with that sentence, writing a_l(x) for the firing strength of leaf l:

          A  score(c) = sum_l a_l(x) * P(c | l)          <- implemented here
          B  score(c) = sum_{l: label(l)=c} a_l(x)       most literal reading
          C  score(c) = sum_{l: label(l)=c} a_l(x)*p(R_l)

        P(c | l) at c = label(l) is exactly Eq. 5's p(R_l), so C is A with the
        minority-class mass discarded and is therefore not a distribution. A is
        used because it is the only one of the three that normalises to a proper
        posterior, and because A, B and C coincide exactly when every leaf is
        pure -- they differ only on impure leaves. That difference is not small:
        over 30 query neighbourhoods A and B disagree on ~10.6% of points (worst
        case 19.4%) and reach 0.968 vs 0.897 fidelity to the teacher.

        This choice is an interpretation, not a result derived from the paper.
        """
        proba_list = []
        for x in X:
            votes = self._predict_sample(x, self.root)
            total = sum(votes.values()) + 1e-12
            # normalize to sum=1
            proba = {cls: votes.get(cls, 0) / total for cls in self.classes_}
            proba_list.append(proba)
        return proba_list


    def predict(self, X):
        proba_list = self.predict_proba(X)
        return np.array([max(p, key=p.get) for p in proba_list])
    

    # -----------------------------
    # RULE EXTRACTION
    # -----------------------------


    def get_linguistic_labels(self, n_partitions):
        if n_partitions == 3:
            return ["Low", "Medium", "High"]
        elif n_partitions == 5:
            return ["Very Low", "Low", "Medium", "High", "Very High"]
        else:
            # Generic fallback
            return [f"Set{i}" for i in range(n_partitions)]


    # def get_global_label(self, feature_idx, local_center):
    #     g_min, g_max = self.global_bounds[feature_idx]
        
    #     # Calculate where this local center sits on the 0 to 1 global scale
    #     # (e.g., 1.2 bar in a 0-30 bar range is ~0.04)
    #     relative_pos = (local_center - g_min) / (g_max - g_min + 1e-12)

    #     if relative_pos < 0.33:
    #         return "Low"
    #     elif relative_pos < 0.66:
    #         return "Medium"
    #     else:
    #         return "High"


    def extract_rules_simple(self, feature_names=None):
        rules = []
        labels_map = self.get_linguistic_labels(self.n_partitions)

        def walk(node, conditions):
            if node.label is not None:
                # Leaf: build rule string
                rule = "IF " + " AND ".join(conditions) if conditions else "IF True"
                rule += f" THEN class = {node.label} [support={node.support:.2f}, confidence={node.confidence:.2f}]"
                rules.append(rule)
                return

            # Internal node: loop over all children
            f = self._fname(node.feature, feature_names)
            for k, child_node in node.children:
                cond = f"{f} IS {labels_map[k]}"
                walk(child_node, conditions + [cond])

        walk(self.root, [])
        return rules
    

    def extract_rules(self, feature_names=None, query=None):
        """With `query`, each rule is annotated with the strength it fires at for
        that point, so the rule explaining the prediction is identifiable."""
        rules = []
        labels_map = self.get_linguistic_labels(self.n_partitions)
        query = None if query is None else np.asarray(query).ravel()

        def walk(node, conditions, strength=1.0):
            """
            Returns:
                dict[label] = list of tuples:
                    (conditions, support, class_mass)
            Mass and support are accumulated separately so that merged sibling
            rules get confidence = sum(mass)/sum(support); averaging the child
            confidences instead over-weights small, pure leaves.
            """
            if node.label is not None:
                return {node.label: [(conditions, node.support,
                                      node.raw_class_masses[node.label], strength)]}

            # Group results from children by their outcome (class label)
            combined_results = {}
            f_name = self._fname(node.feature, feature_names)
            
            if query is None:
                mu = None
            else:
                mu = np.array([gaussian(query[node.feature], c, s)
                               for c, s in zip(node.centers, node.sigmas)])
                mu = mu / (mu.sum() + 1e-12)

            for k, child_node in node.children:
                readable_label = labels_map[k]
                child_strength = strength if mu is None else strength * mu[k]

                # Recurse to get rules from lower levels
                child_rules = walk(child_node, [], child_strength)
                
                for label, paths in child_rules.items():
                    if label not in combined_results:
                        combined_results[label] = {}
                    
                    # Group paths that have the exact same logic after this node
                    for path_conds, supp, mass, fire in paths:
                        path_tuple = tuple(path_conds)
                        if path_tuple not in combined_results[label]:
                            combined_results[label][path_tuple] = {'labels': [], 'supp': 0.0,
                                                                   'mass': 0.0, 'fire': 0.0}

                        # Prevent "(Low OR Low)"
                        if readable_label not in combined_results[label][path_tuple]['labels']:
                            combined_results[label][path_tuple]['labels'].append(readable_label)

                        combined_results[label][path_tuple]['supp'] += supp
                        combined_results[label][path_tuple]['mass'] += mass
                        combined_results[label][path_tuple]['fire'] += fire

            # Build the conditions for the current level
            final_node_rules = {}
            for label, paths in combined_results.items():
                final_node_rules[label] = []
                for path_tail, data in paths.items():
                    # Merge sibling labels: "Low" + "Medium" -> "(Low OR Medium)"
                    if len(data['labels']) == self.n_partitions:
                        merged_label = "ANY" # Technically shouldn't happen after pruning
                    else:
                        merged_label = " OR ".join(data['labels'])
                        if len(data['labels']) > 1:
                            merged_label = f"({merged_label})"
                    
                    current_cond = f"{f_name} IS {merged_label}"
                    new_path = [current_cond] + list(path_tail)
                    final_node_rules[label].append((new_path, data['supp'], data['mass'], data['fire']))
            
            return final_node_rules

        #root_to_leaf_rules= self.extract_rules_simple(feature_names)
        #dot = plot_fuzzy_tree(self.root, feature_names=self.feature_names)
        #dot.render("newfuzzy_tree_pruned", view=True)
        
        # Initial call
        structured_rules = walk(self.root, [])
        if len(structured_rules.keys())==1 and not len(list(structured_rules.values())[0][0][0]):
            # degenerate only root case
            _, supp, mass, _ = list(structured_rules.values())[0][0]
            conf = mass / (supp + 1e-12)
            return [f"IF TRUE THEN class = {self.root.label} [support={supp:.2f}, confidence={conf:.2f}]"]
        # Flatten for output
        for label, rule_list in structured_rules.items():
            for conds, supp, mass, fire in rule_list:
                conf = mass / (supp + 1e-12)
                stats = f"support={supp:.2f}, confidence={conf:.2f}"
                if query is not None:
                    stats += f", fires={fire:.2f}"
                rules.append(f"IF {' AND '.join(conds)} THEN class = {label} [{stats}]")
                
        return rules


    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


    def extract_importance(self, features=None):
        """Support-weighted split importance per feature, normalised to max 1.

        Read off the tree rather than the printed rules: a rule backed by a
        support of 3 should not count as much as one backed by 76, and a
        root-only tree has no splits rather than an unparseable rule string.
        """
        n = self._n_features()
        if features is not None:
            if len(features) < n:
                raise ValueError(f"features has {len(features)} entries, tree was fitted on {n}")
            n = len(features)
        saliency = np.zeros(n)

        def walk(node):
            if node is None or node.label is not None:
                return
            if node.feature < n:
                saliency[node.feature] += node.support or 0.0
            for _, child in node.children:
                walk(child)

        walk(self.root)
        peak = saliency.max()
        return saliency / peak if peak > 0 else saliency

    def plot_feature_memberships(self, feature_name, scaler=None, margin = 3, name=''):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        f_idx = self._findex(feature_name)
        centers, sigmas = self.fuzzy_params[f_idx]
        labels = self.get_linguistic_labels(self.n_partitions)
        
        # define domain from fuzzy params
        if scaler is not None:
            feat_mean = scaler.mean_[f_idx]; feat_scale = scaler.scale_[f_idx]
            centers = (np.array(centers) * feat_scale) + feat_mean
            sigmas = scaler.scale_[f_idx] * np.array(sigmas)

        s = max(sigmas)
        x_min = min(centers) - margin * s
        x_max = max(centers) + margin * s
        x = np.linspace(x_min, x_max, 500)

        plt.figure()
        for c, s, lbl in zip(centers, sigmas, labels):
            mu = np.exp(-0.5 * ((x - c) / s) ** 2)
            plt.plot(x, mu, label=lbl)

        plt.title(feature_name)
        plt.xlabel(feature_name)
        plt.ylabel("Membership")
        plt.legend()
        plt.grid(True)
        #
        #plt.show()
        name += f'_{feature_name}'
        name = name.replace('/', '_')
        plt.savefig(f'./outputs/plots/memberships/{name}.png')
        plt.close()



    def plot_fuzzy_tree(self, root, feature_names=None):
        from graphviz import Digraph
        dot = Digraph(format="png")
        node_id = 0
        node_map = {}

        def get_id(node):
            nonlocal node_id
            if node not in node_map:
                node_map[node] = f"n{node_id}"
                node_id += 1
            return node_map[node]

        def walk(node):
            nid = get_id(node)

            # Leaf
            if not hasattr(node, "children") or not node.children:
                label = f"Leaf\nclass={node.label}"
                dot.node(nid, label, shape="box")
                return

            # Internal node
            fname = self._fname(node.feature, feature_names)
            dot.node(nid, f"{fname}", shape="ellipse")

            for k, child in node.children:
                cid = get_id(child)
                walk(child)
                dot.edge(nid, cid, label=self._set_label(k))

        walk(root)
        return dot



# # =============================
# # Example usage
# # =============================
# X, y = load_iris(return_X_y=True)
# feature_names = [f"x{i}" for i in range(X.shape[1])]

# tree = FuzzyDecisionTree(max_depth=3)

# # ---- ONE LINE TRAIN ----
# tree.fit(X, y)

# # ---- FUZZY RULES AS TEXT ----
# rules = tree.extract_rules(feature_names)

# for i, r in enumerate(rules, 1):
#     print(f"\nRule {i}:")
#     print(r)
