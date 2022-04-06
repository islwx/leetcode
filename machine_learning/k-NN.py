import numpy as np

class Case:
    def __init__(self, feat, label):
        self.feat = feat
        self.label = label

class KDTreeNode:
    def __init__(self, cases=None, parent=None):
        self.meid
        self.cases = cases
        self.median = np.median(self.cases[0][:, dim])
        self.parent = parent
        self._build_sub_node()
    
    @property
    def dim(self):
        if self._dim is None:
            if self.parent is None:
                self._dim = 0
            else:
                self._dim = self.parent.dim + 1
        return self._dim

    
    def _build_sub_node(self):
        is_left = self.cases[0][:, dim]> self.median
        left_cases = [self.cases[0][is_left], self.cases[1][is_left]]
        right_cases = [
            self.cases[0][not is_left], 
            self.cases[1][not is_left]
        ]
        if len(left_cases)>0:
            self.left = self.__class__(left_cases, self)
        else:
            self.left = None
        if len(right_cases)>0:
            self.right = self.__class__(right_cases, self)
        else:
            self.right = None

    
    def search_node(self, feat):
        assert len(feat) == len(self.cases[0][0])
        if feat[self.dim]>self.median:
            if self.left is None:
                return self
            else:
                return self.left.search_node(feat)
        else:
            if self.right is None:
                return self
            else:
                return self.right.search_node(feat)
    

def Lp_distance(a_feat, b_feats, p=0):
    inf = float('inf')
    if isinstance(b_feats, list):
        b_feats = np.asarray(b_feats)
    assert a_feat.shape == b_feats[0].shape
    if p == inf:
        return np.abs((a_feat-b_feats)).max(axis=1)
    elif p == 1:
        return np.abs((a_feat-b_feats)).sum(axis=1)
    else:
        return ((a_feat-b_feats)**p).sum(axis=1)**(1/p)

class KDTree:
    def __init__(self, cases, k=10):
        self.cases = cases
        self.root = KDTreeNode(cases)
        self.p = 2

    def _infer(self, a_feat, node):
        distances = Lp_distance(a_feat, node.cases[0], self.p)

    def infer(self, feat):
        node = self.root.search_node(feat)

        
    

def example():
    feats = np.random.rand(100,100)
    labels = np.random.rand(100).round().astype(int)
    cases = [feats, labels]
    