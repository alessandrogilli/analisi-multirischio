from lib.pca import Pca
from lib.tree import Tree

class Pat:
    def __init__(self, csv_path, n_cluster=0, depth=None):
        self.pca = Pca(csv_path, n_cluster=n_cluster)
        self.tree = Tree(csv_path, n_cluster=n_cluster, depth=depth)