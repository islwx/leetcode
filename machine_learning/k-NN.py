import numpy as np
import pandas as pd
from easydict import EasyDict
from collections import Counter
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import time
from copy import deepcopy
import logging
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)


class LpDistance(object):
    def __init__(self, p):
        self.func=None
        self.p = p
    @property
    def p(self):
        return self._p
    @p.setter
    def p(self, p):
        self._p = p
        inf = float('inf')
        if self.p == inf:
            self.func = lambda a_feat, b_feats: (a_feat-b_feats).abs().max(axis=1)
        elif self.p == 1:
            self.func = lambda a_feat, b_feats: (a_feat-b_feats).abs().sum(axis=1)
        else:
            self.func = lambda a_feat, b_feats: ((a_feat-b_feats)**self.p).sum(axis=1)**(1/self.p)
    def __call__(self, a_feats, b_feats):
        assert a_feats.ndim == b_feats.ndim, (a_feats.ndim, b_feats.ndim)
        distances = []
        for i in range(a_feats.shape[0]):
            a_feat = a_feats[i]
            distances.append(self.func(a_feat, b_feats))
        return distances
    
class KNN:
    def __init__(self, k: int=10):
        self.k: int = k
        self.data: Optional[np.ndarray] = None
        self.target: Optional[np.ndarray] = None
        self.calc_distance = LpDistance(p=2)

    def fit(self, data_: np.ndarray, target_: np.ndarray, train_val=True) -> None:
        if train_val:
            result = []
            trian_data, val_data, train_target, val_target = train_test_split(data_, target_, test_size=0.33, random_state=10)
            self.data=trian_data
            self.target=train_target
            k_max = len(train_target)
            for cur_k in range(1, k_max+1):
                self.k = cur_k
                predictions = self.predict(val_data)
                acc = accuracy_score(predictions, val_target)
                logger.debug([cur_k, acc])
                result.append([cur_k, acc])
            best_k, best_acc=sorted(result,reverse=True, key=lambda x:x[1])[0]
            logger.debug(f"k:{best_k}, acc: {best_acc}")
            self.k = best_k
        self.data = data_
        self.target = target_
    
    def _predict(self, dis):
        distances = dis.argsort()[:self.k]
        counter = Counter([self.target.iloc[index] for index in distances])
        return counter.most_common(1)[0][0]

    def predict(self, data_: np.ndarray) -> int:
        all_distances = self.calc_distance(data_, self.data)
        return [self._predict(distances) for distances in all_distances]


class KDTreeNode:
    def __init__(self, feats=np.empty(0), labels=np.empty(0), parent=None):
        self.median = -1
        self.dim = -1
        self.sub_nodes = []
        self.feats:np.ndarray = feats
        self.labels:np.ndarray = labels
        self.parent = parent
        self._visibled = False
        self.build_sub_node()
    
    @property
    def is_leaf(self):
        return not self.sub_nodes
    
    def build_sub_node(self):
        logger.debug(f"size: {len(self)}, same_label:{(self.labels == self.labels[0]).all()}")
        if len(self)>0 and (self.labels != self.labels[0]).any():
            logger.debug("build sub nodes...")
            std = self.feats.std(axis=0)
            dim = std.argmax()
            median =  np.median(self.feats[:, dim])
            logger.debug(f"sub nodes: std:{std}, dim:{dim}, median:{median}")
            is_left = self.feats[:, dim]>median
            left_feats = self.feats[is_left, :]
            left_labels = self.labels[is_left]
            self.sub_nodes.append(self.__class__(left_feats, left_labels, self))

            right_feats = self.feats[~is_left, :]
            right_labels = self.labels[~is_left]
            self.sub_nodes.append(self.__class__(right_feats, right_labels, self))

            self.dim = dim
            self.median=median
        else:
            logger.debug(f"dont build:{(self.labels == self.labels[0])}")
    
    def __len__(self):
        return len(self.labels)
    
    def deep(self):
        if not self.sub_nodes:
            return 1, 1
        deep0, counts0 = self.sub_nodes[0].deep()
        deep1, counts1 = self.sub_nodes[1].deep()
        return max(deep0, deep1)+1, counts0+counts1+1

    def search_nn_node(self, feat):
        if not self.sub_nodes:
            return self
        if feat[self.dim]>self.median:
            return self.sub_nodes[0].search_nn_node(feat)
        else:
            return self.sub_nodes[1].search_nn_node(feat)
    
    @property
    def visibled(self):
        if self.sub_nodes:
            self._visibled = all(sub_node.visibled for sub_node in self.sub_nodes)
        return self._visibled
    
    @visibled.setter
    def visibled(self, val):
        for sub_node in self.sub_nodes: 
            sub_node.visibled=val
        self._visibled = val


class KDTree:
    def __init__(self, k: int=10):
        self.k: int = k
        self.root = None
        self.calc_distance = LpDistance(p=2)

    def build(self, feats, labels):
        self.root = KDTreeNode(feats, labels)
        logger.critical(self.root.deep())

    def fit(self, feats: np.ndarray, labels: np.ndarray, train_val=False) -> None:
        if train_val:
            result = []
            trian_feats, val_feats, train_labels, val_labels = train_test_split(feats, labels, test_size=0.33, random_state=10)
            self.build(trian_feats, train_labels)
            k_max = len(trian_feats)
            for cur_k in range(1, k_max+1):
                self.k = cur_k
                predictions = self.predict(val_feats)
                acc = accuracy_score(predictions, val_labels)
                logger.debug([cur_k, acc])
                result.append([cur_k, acc])
            best_k, best_acc=sorted(result,reverse=True, key=lambda x:x[1])[0]
            logger.critical(f"k:{best_k}, acc: {best_acc}")
            self.k = best_k
        self.build(feats, labels)

    
    def do_view(self, node, feat, nn_d):
        assert not node.is_leaf
        assert nn_d
        d_max = max(nn_d)
        median_feat = deepcopy(feat)
        median_feat[node.dim]=node.median
        median_feat_distance = self.calc_distance(feat.reshape(1,-1), median_feat.reshape(1,-1))[0][0]
        return median_feat_distance<d_max

    def view_leaf_node(self, node, feat, nn_feat, nn_label, nn_d):
        distances = self.calc_distance(feat.reshape(1,-1), node.feats)[0]
        d_max_ind = np.argmax(nn_d) if len(nn_d)>0 else 0
        d_max = nn_d[d_max_ind] if len(nn_d)>0 else 0
        for feat, label, distance in zip(node.feats, node.labels, distances):
            if len(nn_d)<self.k:
                nn_feat.append(feat)
                nn_label.append(label)
                nn_d.append(distance)
                if distance>=d_max:
                    d_max = distance
                    d_max_ind = -1
            elif distance<d_max:
                nn_feat[d_max_ind]=feat
                nn_label[d_max_ind]=label
                nn_d[d_max_ind]=distance
                d_max_ind = np.argmax(nn_d)
                d_max = nn_d[d_max_ind]
        node.visibled = True

    def breaktrace(self, node, feat, nn_feat, nn_label, nn_d, root=None):
        if not node.visibled:
            if node.is_leaf:
                self.view_leaf_node(node, feat, nn_feat, nn_label, nn_d)
            else:
                if self.do_view(node, feat, nn_d):
                    first_ind = not(feat[node.dim]>node.median)
                    self.breaktrace(node.sub_nodes[first_ind],     feat, nn_feat, nn_label, nn_d, node)
                    self.breaktrace(node.sub_nodes[not first_ind], feat, nn_feat, nn_label, nn_d, node)
                else:
                    logger.debug(f"deep:{node.deep()}")
                node.visibled=True
        if node.parent is not root:
            self.breaktrace(node.parent, feat, nn_feat, nn_label, nn_d, root)

    def _predict(self, feat):
        nn_feat = []
        nn_label = []
        nn_d = []
        self.root.visibled=False
        nn_node = self.root.search_nn_node(feat)
        self.breaktrace(nn_node, feat, nn_feat, nn_label, nn_d, None)
        
        return nn_label

    def predict(self, data_: np.ndarray) -> int:
        predictions = []
        for feat in data_:
            nn_label = self._predict(feat)
            c = Counter(nn_label)
            predictions.append(c.most_common()[0][0])
        return predictions



def example_knn():
    wine = load_wine(return_X_y=True, as_frame=True)
    data: DataFrame = wine[0]
    target: DataFrame = wine[1]
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=1000)
    scalar = MinMaxScaler()
    x_train = scalar.fit_transform(x_train,)
    x_test = scalar.fit_transform(x_test)
    # y_train = y_train.to_numpy()
    # y_test = y_test.to_numpy()
    model = KNN(k=10)
    model.fit(x_train, y_train,train_val=True)
    # model.fit(x_train, y_train,train_val=False)
    a=time.time()
    predictions = model.predict(x_test)
    b=time.time()
    logger.critical(f"knn: {b-a}")
    logger.critical(f"acc: {accuracy_score(predictions, y_test)}")

def example_kd_tree():
    wine = load_wine(return_X_y=True, as_frame=True)
    data: DataFrame = wine[0]
    target: DataFrame = wine[1]
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=1000)
    scalar = MinMaxScaler()
    x_train = scalar.fit_transform(x_train)
    x_test = scalar.fit_transform(x_test)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    model = KDTree(k=10)
    # model.fit(x_train, y_train, train_val=False)
    model.fit(x_train, y_train,train_val=True)
    a=time.time()
    predictions = model.predict(x_test)
    b=time.time()
    logger.critical(f"kd-tree: {b-a}")
    logger.critical(f"acc: {accuracy_score(predictions, y_test)}")


def example_kd_tree_debug():
    wine = load_wine(return_X_y=True, as_frame=True)
    data: DataFrame = wine[0]
    target: DataFrame = wine[1]
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=1000)
    scalar = MinMaxScaler()
    x_train = scalar.fit_transform(x_train)
    x_test = scalar.fit_transform(x_test)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    model = KDTree(k=10)
    # model.fit(x_train, y_train, train_val=False)
    model.fit(x_train, y_train,train_val=True)
    
    from line_profiler import LineProfiler
    lp = LineProfiler()
    lp.add_function(model.breaktrace)   # 监控子函数
    lp.add_function(model.view_leaf_node)   # 监控子函数
    lp.add_function(model.do_view)   # 监控子函数
    lp.runcall(model.predict, x_test)     # 启动监控主体，并输入参数
    lp.print_stats()                  # 打印
    

if __name__ == '__main__':
    # example_knn()
    # example_kd_tree()
    example_kd_tree_debug()