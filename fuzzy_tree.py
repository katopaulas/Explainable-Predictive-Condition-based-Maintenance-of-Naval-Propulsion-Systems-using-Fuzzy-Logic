import numpy as np

def plot_fuzzy_tree(root, feature_names=None):
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
        f = node.feature
        fname = feature_names[f] if feature_names is not None else f"x[{f}]"
        dot.node(nid, f"{fname}", shape="ellipse")

        for edge_label, child in node.children:
            cid = get_id(child)
            walk(child)
            dot.edge(nid, cid, label=edge_label)

    walk(root)
    return dot

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
        support     : How strongly this rule fires in the data
        confidence  : How often the rule is correct when it fires
        centers     : Fuzzy set centers for this feature split
        sigmas      : Fuzzy set spreads for this feature split
        e.g   RULE A     IF ...  THEN class = 2
                                 [support=23.7, confidence=0.91]
    '''

    def __init__(self, feature=None, children=None, label=None,
                 support=None, confidence=None, centers=None, sigmas=None,
                 class_masses=None):
        self.feature = feature
        self.children = children  # [(label, node), ...] or None if leaf
        self.label = label
        self.support = support
        self.confidence = confidence
        self.centers = centers   # centers of fuzzy sets corresponding to children
        self.sigmas  = sigmas     
        self.class_masses = class_masses
        


# -----------------------------
# Fuzzy Decision Tree
# -----------------------------
class FuzzyDecisionTree:
    """
        n_partitions = number of fuzzy sets per feature (e.g., 3 → Low/Medium/High)
    """
    def __init__(self, max_depth=3, min_samples=10, n_partitions=3,
                 min_support=10, min_confidence=0.5,feature_names=[],
                 X_global=None, verbose=False):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_partitions = n_partitions
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.feature_names=feature_names
        self.root = None
        self.classes_ = None
        self.verbose = verbose

        self.min_gain=5e-3; self.min_gain_ratio=0.01
        self.banned_features = ['lever_position', 'ship_speed(v)']


    def get_fuzzy_sets_local(self, x, w):
        c = np.average(x, weights=w)
        s = np.sqrt(np.average((x - c)**2, weights=w)) + 1e-6
        centers = [c - s, c, c + s]
        sigmas = [s, s, s]
        return centers, sigmas

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
        # Assuming we store global_bounds as {idx: (min, max)}
        #c_min, c_max = self.global_bounds[feature_idx]

        n_sets = self.n_partitions
        c_min, c_max = X_column.min(), X_column.max() # this is local
        if n_sets == 1:
            centers = [(c_min + c_max)/2]
            sigmas = [(c_max - c_min)/2 + 1e-6]
        else:
            centers = np.linspace(c_min, c_max, n_sets)
            sigma = (c_max - c_min) / (2 * (n_sets - 1)) + 1e-6
            sigmas = [sigma] * n_sets

        return centers, np.array(sigmas)
        

    def fit(self, X, y):    
        self.classes_, counts = np.unique(y, return_counts=True)
        # Weight = Total Samples / (Number of Classes * Class Count)
        class_weights = {class_id: len(y) / (len(self.classes_) * count) for class_id, count in zip(self.classes_, counts)}
        w = np.array([class_weights[class_id] for class_id in y])
        #w = np.ones(len(y))
        #self.classes_ = np.unique(y)
        self.root = self._build(X, y, w, depth=0)
        n_nodes = count_nodes(self.root)
        

        # Automatically prune after building
        self.prune(min_support=self.min_support, min_confidence=self.min_confidence)
        n_nodes2 = count_nodes(self.root)

        if self.verbose:
            print(f'Trained tree with {n_nodes} nodes')
            if n_nodes != n_nodes2:
                print(f'From {n_nodes} nodes, pruned to {n_nodes2} nodes')
            print()


    def _build(self, X, y, w, depth, features_used=None):
        n_features = X.shape[1]
        class_masses = {label: float(w[y == label].sum()) for label in self.classes_}
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
                if f in features_used or self.feature_names[f] in self.banned_features: # avoid immediate reuse
                    continue
                f_name = self.feature_names[f]
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
               
            
            # Prepare children nodes
            linguistic_labels = [f"Set{i}" for i in range(self.n_partitions)]
            children_nodes = []
            for i in range(self.n_partitions):
                #mask = (winner == i)
                #if not np.any(mask):
                #    continue
                #w_child = w * mu_sets[i]*mask

                w_child = w * mu_sets[i]
                if w_child.sum() > 1e-6:
                    new_features_used = features_used + [best_f]
                    child_node = self._build(X, y, w_child, depth + 1, new_features_used)
                    children_nodes.append((linguistic_labels[i], child_node))

            # Degenerate case: if no children, make leaf
            if len(children_nodes) == 0:
                stop_condition = True


        # ---------- LEAF ----------(?)
        if stop_condition:
            label = weighted_majority(y, w)
            support = w.sum(); #support = raw_support /len(y)
            # If not dominant fuzzy : Confidence represents the fuzzy proportion of class 
            # density in this hyper-volume.
            confidence = class_masses[label] / (support + 1e-12)
            return FuzzyNode(label=label, support=support, confidence=confidence,
                             class_masses=class_masses)

        # ---------- INTERNAL NODE ----------
        return FuzzyNode(
            feature=best_f,
            children=children_nodes,
            centers=best_centers,
            sigmas=best_sigmas,
            class_masses=class_masses
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

        def _feature_name(feature):
            if feature is None:
                return None
            if len(self.feature_names) > feature:
                return self.feature_names[feature]
            return f"Feature_{feature}"

        def _collapse_to_leaf(node, path, reason, weak_children=None):
            masses = node.class_masses
            if not masses:
                raise RuntimeError("Cannot prune a node without class masses")
            split_feature = _feature_name(node.feature)
            labels_before = [child.label for _, child in node.children] if node.children else []
            node.label = max(masses, key=masses.get)
            node.support = sum(masses.values())
            node.confidence = masses[node.label] / (node.support + 1e-12)
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
                (name, _prune_node(child, path + (f"{_feature_name(node.feature)}:{name}",)))
                for name, child in node.children
            ]
            node.children = [(n, c) for n, c in node.children if c is not None]

            if not node.children:
                return _collapse_to_leaf(node, path, "no_children")

            all_children_are_leaves = all(child.label is not None for _, child in node.children)
            if not all_children_are_leaves:
                return node

            labels = [child.label for _, child in node.children]  
            low_support_children = []
            for set_name, child in node.children:
                child_report = {
                    "path": tuple(path + (f"{_feature_name(node.feature)}:{set_name}",)),
                    "set": set_name,
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


    def myprune(self, min_support=1.0, min_confidence=0.5):
        """
        Recursively prune leaves that do not meet thresholds.
        Converts them into leaves with majority class if needed.
        """

        def _prune_node(node):
            if node.label is not None:
                # # Leaf: check thresholds
                # if node.support < min_support or node.confidence < min_confidence:
                #     # weak leaf
                #     node.confidence = 0.0
                return node

            # Internal node: recurse on children
            pruned_children = []
            for set_name, child in node.children:
                pruned_child = _prune_node(child)
                pruned_children.append((set_name, pruned_child))
            node.children = pruned_children

            # SIBLING COLLAPSING: Check if all children lead to the same result
            all_children_are_leaves = all(child.label is not None for _, child in node.children)
            if all_children_are_leaves:
                labels = [child.label for _, child in node.children]
                unique_labels = set(labels)
                # If all siblings have the same label, the split on 'node.feature' was redundant
                if len(unique_labels) == 1:
                    node.label = labels[0]
                    node.support = sum(child.support for _, child in node.children)
                    if node.support > 1e-12:
                        node.confidence = sum(c.confidence * c.support for _, c in node.children) / node.support
                    else:
                        node.confidence = 0.0
                    # This can squash if a child was very weak
                    #node.confidence = np.mean([child.confidence for _, child in node.children])
                    node.children = None
                    return node
                
                total_s = sum(child.support for _, child in node.children)
                avg_conf = sum(c.confidence * c.support for _, c in node.children) / (total_s + 1e-12)
                if total_s < min_support or avg_conf < min_confidence:
                    # Collapse into majority class leaf
                    total_supports = {child.label: 0 for _, child in node.children}
                    for _, child in node.children:
                        total_supports[child.label] += child.support
                    
                    node.label = max(total_supports, key=total_supports.get)
                    node.support = total_s
                    node.confidence = avg_conf
                    node.children = None
                    return node

            return node

        if self.root:
            self.root = _prune_node(self.root)


    def _predict_sample(self, x, node, strength=1.0):
        """
        Recursively compute class vote for a single sample x.
        strength = current membership weight reaching this node
        """
        if node.label is not None:
            # Leaf: vote = strength * confidence
            return {node.label: strength * node.confidence}

            # Compute memberships for this feature
        mu = np.array([
            gaussian(x[node.feature], c, s)
            for c, s in zip(node.centers, node.sigmas)
        ])
        mu = mu / (mu.sum() + 1e-12)
        # Internal node: loop over all children
        votes = {}
        for i, (set_name, child_node) in enumerate(node.children):
            child_strength = strength * mu[i]
            if child_strength > 1e-6:
                child_votes = self._predict_sample(x, child_node, child_strength)
                for k, v in child_votes.items():
                    votes[k] = votes.get(k, 0) + v
        return votes
    

    def predict_proba(self, X):
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


    def extract_rules_simple(self, feature_names=[]):
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
            f = feature_names[node.feature]
            for set_name, child_node in node.children:
                # Convert Set0 / Set1 / Set2 to human-readable
                try:
                    idx = int(set_name.replace("Set", ""))
                    readable_label = labels_map[idx]
                except:
                    readable_label = set_name  # fallback
                cond = f"{f} IS {readable_label}"
                walk(child_node, conditions + [cond])

        walk(self.root, [])
        return rules
    

    def extract_rules(self, feature_names=[]):
        rules = []
        # COMMENT FOR GLOBAL LABELS
        labels_map = self.get_linguistic_labels(self.n_partitions)

        def walk(node, conditions):
            """
            Returns:
                dict[label] = list of tuples:
                    (conditions, support, confidence)
            """
            if node.label is not None:
                return {node.label: [(conditions, node.support, node.confidence)]}

            # Group results from children by their outcome (class label)
            combined_results = {}
            f_name = feature_names[node.feature]
            
            for i, (set_name, child_node) in enumerate(node.children):
                
                # #UN/COMMENT FOR GLOBAL LABELS
                # local_center = node.centers[i]
                # readable_label = self.get_global_label(node.feature, local_center)
                
                # COMMENT 2 below lines for GLOBAL LABELS
                idx = int(set_name.replace("Set", ""))
                readable_label = labels_map[idx]
                
                # Recurse to get rules from lower levels
                child_rules = walk(child_node, []) # Start fresh conditions for sub-path
                
                for label, paths in child_rules.items():
                    if label not in combined_results:
                        combined_results[label] = {}
                    
                    # Group paths that have the exact same logic after this node
                    for path_conds, supp, conf in paths:
                        path_tuple = tuple(path_conds)
                        if path_tuple not in combined_results[label]:
                            combined_results[label][path_tuple] = {'labels': [], 'supp': 0, 'conf': []}
                        
                        # Use set() or a check to prevent "(Low OR Low)"
                        if readable_label not in combined_results[label][path_tuple]['labels']:
                            combined_results[label][path_tuple]['labels'].append(readable_label)
                            
                        #combined_results[label][path_tuple]['labels'].append(readable_label)
                        combined_results[label][path_tuple]['supp'] += supp
                        combined_results[label][path_tuple]['conf'].append(conf)

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
                    avg_conf = sum(data['conf']) / len(data['conf'])
                    final_node_rules[label].append((new_path, data['supp'], avg_conf))
            
            return final_node_rules

        #root_to_leaf_rules= self.extract_rules_simple(feature_names)
        #dot = plot_fuzzy_tree(self.root, feature_names=self.feature_names)
        #dot.render("newfuzzy_tree_pruned", view=True)
        
        # Initial call
        structured_rules = walk(self.root, [])
        if len(structured_rules.keys())==1 and not len(list(structured_rules.values())[0][0][0]):
            # degenerate only root case
            rules, supp, conf = list(structured_rules.values())[0][0]
            return [f"IF TRUE THEN class = {self.root.label} [support={supp:.2f}, confidence={conf:.2f}]"]
        # Flatten for output
        for label, rule_list in structured_rules.items():
            for conds, supp, conf in rule_list:
                rule_str = f"IF {' AND '.join(conds)} THEN class = {label} [support={supp:.2f}, confidence={conf:.2f}]"
                rules.append(rule_str)
                
        return rules


    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


    def extract_importance(self, human_rules, features):
        saliency = np.zeros(len(features))
        hash_ = {key:i for i,key in enumerate(features)}
        for rule in human_rules:
            conditions = rule.split(' AND ')
            for c in conditions:
                if c.startswith('IF'):
                    c = c[3:]
                feature = c.split(' IS ')[0]
                saliency[hash_[feature]]+=1
        return saliency/max(saliency)


    def plot_feature_memberships(self, feature_name, scaler=None, margin = 3, name=''):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        f_idx = self.feature_names.get_loc(feature_name)
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
            f = node.feature
            fname = feature_names[f] if feature_names is not None else f"x[{f}]"
            dot.node(nid, f"{fname}", shape="ellipse")

            for edge_label, child in node.children:
                cid = get_id(child)
                walk(child)
                dot.edge(nid, cid, label=edge_label)

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
