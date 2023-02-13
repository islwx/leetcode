import sklearn.datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class PerceptronOperator(object):
    def __init__(self, num_weight):
        self.num_weight = num_weight
        self.weight = np.zeros(num_weight, dtype=np.float64)
        self.bias = 0
        self.lr = 0.01
    
    def forward(self, x):
        return np.dot(x, self.weight) + self.bias

    def fit(self, x, y):
        y_ = self(x)
        if y_*y <= 0:
            self.weight += self.lr * np.dot(y, x)
            self.bias += self.lr * y
            return True
        else:
            return False
            
    
    def __call__(self, x):
        return self.forward(x)



class NaivaBayers(object):
    def __init__(self):
        pass
    
    def build(self, inputs, targets):
        pass

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, target):
        self._label.fill(-1)
        self._label[target]=1

    def fit(self, inputs, targets):
        pass

    def forward(self, inp):
        pass

    def predict(self, inp):
        pass
        



def load_dataset():
    dataset = sklearn.datasets.load_iris(return_X_y=True, as_frame=True)
    return dataset

def example():
    dataset = load_dataset()
    data: DataFrame = dataset[0]
    target: DataFrame = dataset[1]
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=1000)
    scalar = MinMaxScaler()
    onehoter = OneHotEncoder()
    x_train = scalar.fit_transform(x_train)
    x_test = scalar.fit_transform(x_test)
    y_train = onehoter.fit_transform(y_train)
    y_test = y_test.to_numpy()
    model = NaivaBayers()
    model.fit(x_train, y_train)
    predictions = [model.predict(x) for x in x_test]
    print(accuracy_score(predictions, y_test))

if __name__ == '__main__':
    example()