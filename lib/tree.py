import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree


class Tree:
    def __init__(self, url, n_cluster=0, depth=None, verbose=0):
        self.depth = depth
        self.url = url
        self.targets = []
        self.verbose = verbose

        if n_cluster == 0:
            try:
                self.n_cluster = int(self.url[self.url.rfind('_') + 1:-4])  # ricavo numero di cluster da nome del csv
            except ValueError:
                self.n_cluster = int(input("Insert number of clusters: "))
        else:
            self.n_cluster = n_cluster

        for n in range(self.n_cluster):
            self.targets.append('cluster' + str(n))

        self.df = pd.read_csv(self.url,
                              names=['DZCOM', 'DENSPOP', 'AGMAX_50', 'IDR_POPP3', 'IDR_POPP2', 'IDR_POPP1',
                                     'IDR_AREAP1',
                                     'IDR_AREAP2', 'IDR_AREAP3', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E19',
                                     'E20',
                                     'E30', 'E31', 'Cluster'])

        self.features = ['DENSPOP', 'AGMAX_50', 'IDR_POPP3', 'IDR_POPP2', 'IDR_POPP1', 'IDR_AREAP1', 'IDR_AREAP2',
                         'IDR_AREAP3', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E19', 'E20', 'E30',
                         'E31']  # Identifica gli attributi

        x = self.df.loc[:, self.features].values
        y = self.df['Cluster']

        self.clf = DecisionTreeClassifier(max_depth=depth).fit(x, y)

    def plot(self, save=False):

        plt.figure(figsize=(22, 10))

        plot_tree(self.clf, filled=True, fontsize=8, feature_names=self.features, class_names=self.targets,
                  node_ids=True)
        if save:
            filename = self.url.replace("CSV/", "")[:-4] + "_depth" + str(self.depth) + ".png"
            self.saving("img",filename)

    def saving(self, t, name, output=""):
        if t == "img":
            path = "img_tree"
            try:
                os.mkdir(path)
            except FileExistsError:
                if (self.verbose >= 1):
                    print("%s exists yet" % path)
            plt.savefig(path + "/" + name)
        elif t == "txt":
            path = "txt_tree"
            try:
                os.mkdir(path)
            except FileExistsError:
                if (self.verbose >= 1):
                    print("%s exists yet" % path)

            if self.depth is None:
                self.depth = "Auto"
            file = open(path + "/" + name, 'w')
            print(output, file=file)
            file.close()

    def description(self, save=False):
        n_nodes = self.clf.tree_.node_count
        children_left = self.clf.tree_.children_left
        children_right = self.clf.tree_.children_right
        feature = self.clf.tree_.feature
        threshold = self.clf.tree_.threshold
        val = self.clf.tree_.value.tolist()
        classidx = []
        for v in val:
            classidx.append(v[0].index(max(v[0])))

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True

        output = "The binary tree structure has {n} nodes and has the following tree structure:\n".format(n=n_nodes)
        for i in range(n_nodes):
            if is_leaves[i]:
                output += "{space}node={node} is a leaf node and belongs to {cls}.\n".format(
                    space=node_depth[i] * "\t", node=i, cls=self.targets[classidx[i]])
            else:
                output += "{space}node={node} is a split node: " \
                          "go to node {left} if {feature} <= {threshold} else to node {right}.\n".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=self.features[feature[i]],
                    threshold=threshold[i],
                    right=children_right[i])
        print(output)

        if save:
            filename = self.url.replace("CSV/", "")[:-4] + "_depth" + str(self.depth) + "_description_tree.txt"
            self.saving(t="txt",name=filename,output=output)

    def text(self, save=False):
        from sklearn.tree import export_text
        self.r = export_text(self.clf, feature_names=self.features)
        print(self.r)

        if save:
            filename = self.url.replace("CSV/", "")[:-4] + "_depth" + str(self.depth) + "_tree.txt"
            self.saving(t="txt", name=filename, output=self.r)

    def get_rules(self):
        from sklearn.tree import _tree
        feature_names = self.features
        class_names = self.targets
        tree_ = self.clf.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        paths = []
        path = []

        def recurse(node, path, paths):

            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                p1, p2 = list(path), list(path)
                p1 += [f"({name} <= {np.round(threshold, 3)})"]
                recurse(tree_.children_left[node], p1, paths)
                p2 += [f"({name} > {np.round(threshold, 3)})"]
                recurse(tree_.children_right[node], p2, paths)
            else:
                path += [(tree_.value[node], tree_.n_node_samples[node])]
                paths += [path]

        recurse(0, path, paths)

        # sort by samples count
        samples_count = [p[-1][1] for p in paths]
        ii = list(np.argsort(samples_count))
        paths = [paths[i] for i in reversed(ii)]

        rules = []
        for path in paths:
            rule = "if "

            for p in path[:-1]:
                if rule != "if ":
                    rule += " and "
                rule += str(p)
            rule += " then "
            if class_names is None:
                rule += "response: " + str(np.round(path[-1][0][0][0], 3))
            else:
                classes = path[-1][0][0]
                l = np.argmax(classes)
                rule += f"class: {class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
            rule += f" | based on {path[-1][1]:,} samples"
            rules += [rule]

        return rules

    def rules(self, save=False):
        rules = self.get_rules()
        output=""
        for r in rules:
            output += r + "\n" #print(r)
        print(output)

        if save:
            filename = self.url.replace("CSV/", "")[:-4] + "_depth" + str(self.depth) + "_rules.txt"
            self.saving(t="txt",name=filename,output=output)

    def show(self):
        plt.show()
