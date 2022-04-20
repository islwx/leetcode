import sklearn.datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np


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



class Perceptron(object):
    def __init__(self):
        self.num_weight:int = None
        self.num_class:int = None
        self.operators:list = None
        self._label:np.ndarray = None
    
    def build(self, inputs, targets):
        num_weight = inputs.shape[1]
        num_class = targets.max()+1
        self.num_weight = num_weight
        self.num_class = num_class
        self.operators = [PerceptronOperator(num_weight) for _ in range(num_class)]
        self._label = np.ones(num_class)

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, target):
        self._label.fill(-1)
        self._label[target]=1

    def fit(self, inputs, targets):
        self.build(inputs, targets)
        has_wrong = True
        last_cout = len(targets)
        while has_wrong:
            wrong_count = 0
            for inp, target in zip(inputs, targets):
                self.label = target
                if self.predict(inp) != target:
                    [self.operators[i].fit(inp, self.label[i]) for i in range(self.num_class)]
                    wrong_count+=1
            if wrong_count < 2:
                has_wrong = False
            if wrong_count > last_cout:
                s = 1.01
            else:
                s = 0.99
            for opt in self.operators:
                opt.lr*=s
            last_cout = wrong_count
            print(wrong_count)

    def forward(self, inp):
        output =[opt(inp) for opt in self.operators]
        return output

    def predict(self, inp):
        return np.argmax(self.forward(inp))
        


def load_dataset():
    dataset = sklearn.datasets.load_iris(return_X_y=True, as_frame=True)
    return dataset

def example():
    dataset = load_dataset()
    data: DataFrame = dataset[0]
    target: DataFrame = dataset[1]
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=1000)
    scalar = MinMaxScaler()
    x_train = scalar.fit_transform(x_train)
    x_test = scalar.fit_transform(x_test)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    model = Perceptron()
    model.fit(x_train, y_train)
    predictions = [model.predict(x) for x in x_test]
    print(accuracy_score(predictions, y_test))

if __name__ == '__main__':
    example()