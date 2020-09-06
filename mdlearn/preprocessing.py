""" Preprocessing
"""

import sklearn.model_selection
import sklearn.preprocessing
import numpy as np


def separate(array, frac):
    n = int(round(len(array) * frac))
    train_array = np.random.choice(array, n, replace=False)
    test_array = np.array([i for i in array if i not in train_array], dtype=int)
    return train_array, test_array


class Selector:
    """ OOP Selector
    """

    def __init__(self, *data):
        """ data: np.array/pandas.df
        """
        self.train_index = None
        self.valid_index = None
        self.test_index = None

        self.kfold = None
        self.kfold_train_indexes = None
        self.kfold_valid_indexes = None

        for d in data[1:]:
            assert len(d) == len(data[0]), 'Data does not have same length'

        self.data = tuple(np.array(d) for d in data)
        self.length = len(self.data[0])
        self.selected_num = np.arange(self.length)

    def partition(self, training_size, validation_size):
        """ Perform 3-partition (training + validating + testing)
        """

        self.train_index = np.zeros(self.length, dtype=bool)
        self.valid_index = np.zeros(self.length, dtype=bool)
        self.test_index = np.zeros(self.length, dtype=bool)

        train_valid_num, test_num = separate(self.selected_num, training_size + validation_size)
        train_num, valid_num = separate(train_valid_num, training_size / (training_size + validation_size))
        self.train_index[train_num] = True
        self.valid_index[valid_num] = True
        self.test_index[test_num] = True

    def kfold_partition(self, train_valid_size, fold=5):
        self.kfold = fold
        self.kfold_train_indexes = [np.ones(self.length, dtype=bool) for i in range(fold)]
        self.kfold_valid_indexes = [np.zeros(self.length, dtype=bool) for i in range(fold)]
        self.test_index = np.zeros(self.length, dtype=bool)

        if train_valid_size < 1:
            train_valid_num, test_num = separate(self.selected_num, train_valid_size)
        else:
            train_valid_num = self.selected_num[:]
            test_num = np.array([], dtype=int)
        self.test_index[test_num] = True

        for k in range(fold):
            valid_num, train_valid_num = separate(train_valid_num, 1 / (fold - k))
            self.kfold_valid_indexes[k][valid_num] = True
            self.kfold_train_indexes[k][valid_num] = False
            self.kfold_train_indexes[k][test_num] = False

    def kfold_use(self, n):
        self.train_index = self.kfold_train_indexes[n]
        self.valid_index = self.kfold_valid_indexes[n]

    def training_set(self):
        idx = self.train_index
        return (d[idx] for d in self.data) if len(self.data) > 1 else self.data[0][idx]

    def validation_set(self):
        idx = self.valid_index
        return (d[idx] for d in self.data) if len(self.data) > 1 else self.data[0][idx]

    def test_set(self):
        idx = self.test_index
        return (d[idx] for d in self.data) if len(self.data) > 1 else self.data[0][idx]

    def train_valid_set(self):
        idx = np.logical_not(self.test_index)
        return (d[idx] for d in self.data) if len(self.data) > 1 else self.data[0][idx]

    def load(self, filename):
        with open(filename, 'r') as f_cache:
            line1 = f_cache.readline().rstrip('\n')
            line2 = f_cache.readline().rstrip('\n')
            line3 = f_cache.readline().rstrip('\n')

            self.train_index = np.array(list(map(int, line1)), dtype=bool)
            self.valid_index = np.array(list(map(int, line2)), dtype=bool)
            self.test_index = np.array(list(map(int, line3)), dtype=bool)

    def save(self, filename):
        with open(filename, 'w') as f_cache:
            f_cache.write(''.join(map(str, self.train_index.astype(int))))
            f_cache.write('\n')
            f_cache.write(''.join(map(str, self.valid_index.astype(int))))
            f_cache.write('\n')
            f_cache.write(''.join(map(str, self.test_index.astype(int))))
            f_cache.write('\n')


class Scaler:
    """ StandardScaler that can save
    """

    def __init__(self, *args, **kwargs):
        self.scaler_ = sklearn.preprocessing.StandardScaler(*args, **kwargs)

    def fit(self, *args):
        return self.scaler_.fit(*args)

    def transform(self, *args):
        return self.scaler_.transform(*args)

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write(','.join(map(str, self.scaler_.mean_)) + '\n')
            f.write(','.join(map(str, self.scaler_.var_)) + '\n')
            f.write(','.join(map(str, self.scaler_.scale_)) + '\n')

    def load(self, filename):
        self.scaler_.scale_ = 1
        with open(filename, 'r') as f:
            line1 = f.readline().strip('\n')
            line2 = f.readline().strip('\n')
            line3 = f.readline().strip('\n')

            self.scaler_.mean_ = np.array(list(map(float, line1.split(','))))
            self.scaler_.var_ = np.array(list(map(float, line2.split(','))))
            self.scaler_.scale_ = np.array(list(map(float, line3.split(','))))
            assert len(self.scaler_.mean_) == len(self.scaler_.var_)
