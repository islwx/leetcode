import numpy as np
import pandas as pd
from easydict import EasyDict
from collections import Counter

class Case:
    def __init__(self, feat, label, d=float('-inf')):
        self.feat = feat
        self.label = label
        self.d=d

class KDTreeNode:
    def __init__(self, cases, medians, parent=None):
        self._dim = None
        self.parent = parent
        self.cases = cases
        self.median = medians[self.dim]
        self._build_sub_node(medians)
    
    @property
    def dim(self):
        if self._dim is None:
            if self.parent is None:
                self._dim = 0
            else:
                self._dim = self.parent.dim + 1
        return self._dim

    @staticmethod
    def split_case(iswhere, cases):
        new_cases = EasyDict()
        for key in cases:
            new_cases[key] = cases[key][iswhere]
        
            
        return new_cases
            

    def _build_sub_node(self, medians):
        if self.dim==medians.size-1:
            self.left =None
            self.right=None
        else:
            is_left = self.cases.feat[:, self.dim]> self.median    
            left_cases = self.split_case(is_left, self.cases)
            right_cases = self.split_case(~is_left, self.cases)

            if len(left_cases.feat)>0:
                self.left = self.__class__(left_cases, medians, self)
            else:
                self.left = None
            if len(right_cases.feat)>0:
                self.right = self.__class__(right_cases, medians, self)
            else:
                self.right = None

    
    def search_nn_node(self, feat):
        assert feat.size == self.cases.feat[0].size,(feat.size, self.cases.feat[0].size)
        if feat[self.dim]>self.median:
            if self.left is None:
                return self
            else:
                return self.left.search_nn_node(feat)
        else:
            if self.right is None:
                return self
            else:
                return self.right.search_nn_node(feat)
    
        

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
        self.medians = np.median(self.cases.feat, axis=0)
        self.root = KDTreeNode(self.cases, self.medians, None)
        self.p = 2
        self.k = 10
        
    def fit_k(self, val_cases):
        root_nn_node = self.root.search_nn_node(feat)
        cout = 0
        k_right_count = {}
        for i in range(len(val_cases.label)):
            feat = val_cases.feat[i]
            label = val_cases.label[i]
            nn_cases = self._infer_nn_cases(feat, root_nn_node)
            cur_nn_cases 
            for nn_cases in sortd(nn_cases, key=lambda cases:case.d):
                pass


            
            

    def _infer_nn_cases(self, feat, node, nn_cases=None):
        if nn_cases is None:
            nn_cases = EasyDict({
                'feat':[],
                'label':[],
                'd':[],
            })
        max_d_for_case = max(nn_cases.d) if len(nn_cases.d)>0 else float("inf")
        distances = Lp_distance(feat, node.cases.feat, self.p)

        for i, d in enumerate(distances):
            if d >= max_d_for_case:
                continue
            if len(nn_cases.d)<self.k:
                nn_cases.feat.append(self.cases.feat[i])
                nn_cases.label.append(self.cases.label[i])
                nn_cases.d.append(d)
            else:
                index = np.argmax(nn_cases.d)
                nn_cases.feat[index] = self.cases.feat[i]
                nn_cases.label[index] = self.cases.label[i]
                nn_cases.d[index] = d
        if node.parent is not None:
            self._infer_nn_cases(feat, node.parent, nn_cases)
        return nn_cases

    def infer(self, feat):
        root_nn_node = self.root.search_nn_node(feat)
        nn_cases = self._infer_nn_cases(feat, root_nn_node, None)
        c=Counter(nn_cases.label)
        return c.most_common(1)[0][0]
        
    

def example():
    num_case = 50
    num_axis = 10
    feats = np.random.rand(num_case,num_axis)
    feat = np.random.rand(num_axis)
    labels = (np.random.rand(num_case)*3).round().astype(int)
    cases = EasyDict({
        'feat':feats,
        'label':labels,
        # 'd':np.full_like(feats, float('inf'))
    })
    # cases = pd.DataFrame(data,columns=['feat','label','d'])
    kdt = KDTree(cases)
    label = kdt.infer(feat)
    print(label)

if __name__ == '__main__':
    example()