# coding: utf-8

import numpy as np
from itertools import groupby


class KNN():
    def __init__(self, fname_data, fname_test, class_position, skip_column=None):
        """
            class_position indicates position of class argument
            in dataset
        """

        self.raw_data = self._get_raw_data(fname_data)
        self.class_position = self._get_class_position(class_position, skip_column)
        self.classification = self._find_classes()
        self.all_data = self._load_data(fname_data, skip_column)
        self.data = self.get_train_data()
        self.data_test = self._load_data(fname_test, skip_column)
        #self.data_test = self.get_test_data()

    def _get_class_position(self, class_position, skip_column):
        if skip_column:
            if skip_column < class_position:
                return class_position - 1
        return class_position

    def _get_raw_data(self, fname):
        rdata = []
        with open(fname) as f:
            rdata = [line.split(',') for line in f]
        return rdata

    def _find_classes(self):
        classes = []
        pos = self.class_position
        for vector in self.raw_data:
            print(vector)
            print(pos)
            if vector[pos] not in classes:
                classes.append(str(vector[pos].strip()))

        classification = dict(zip(classes, range(0, len(classes))))

        return classification

    def _load_data(self, fname, skip_column):
        with open(fname) as f:
            m = np.loadtxt(
                f, delimiter=',', converters={
                self.class_position: lambda x: self.classification[x.decode()]})
            if skip_column:
                m = np.delete(m, skip_column, 1)

            return m


    def get_test_data(self):
        length_test = int(0.3 * len(self.all_data))
        return self.all_data[np.random.choice(self.all_data.shape[0], length_test)]

    def get_train_data(self):
        length_train = int(0.7 * len(self.all_data))
        return self.all_data[np.random.choice(self.all_data.shape[0], length_train)]

    def _delta(self, a, b):
        return 1.0 if a == b else 0.0

    def _get_distance(self, v_1, v_2):
        return np.linalg.norm(v_1 - v_2)

    def _most_commom(self, l):
        l = sorted(l)
        l = [list(group) for key, group in groupby(l)]
        frequency = [(len(i), i[0]) for i in l]

        return max(frequency, key=lambda x:x[0])[1]

    def get_training_set(self):
        return self.data

    def get_testing_set(self):
        return self.data_test

    def get_k_nearest(self, vector, k):
        distances = [
            (train_vector, np.linalg.norm(train_vector - vector))
            for train_vector in self.data
        ]
        k_nearest = sorted(distances, key=lambda x: x[1])[:k]

        return k_nearest

    def get_k_nearest_test(self, k):
        k_nearest = []
        for item in self.data_test:
            k_nearest.append(self.get_k_nearest(item, k))

        return k_nearest

    def accuracy_rate(self, result):
        test = [vector[self.class_position] for vector in self.data_test]
        matches = len([i for i, j in zip(result, test) if i == j])
        total = len(self.data_test)
        rate = float(matches)/total

        return rate

    def solve(self, k):
        k_nearest_test = self.get_k_nearest_test(k)

        result = []
        for tuples in k_nearest_test:
            classes = []
            for vector, distance in tuples:
                classes.append(vector[self.class_position])
            result.append(self._most_commom(classes))

        rate = self.accuracy_rate(result)
        return rate


class KNNWeighted(KNN):

    def argmax(self, k_nearest, v_test):
        args_class = {}
        for key, klass in self.classification.items():
            args_list = [] 
            for v, distance in k_nearest:
                if self._get_distance(v, v_test) == 0:
                    return v[self.class_position]
                arg = self._delta(klass, v[self.class_position])
                arg = arg / (self._get_distance(v, v_test) ** 2)
                args_list.append(arg)
            args_class[key] = sum(args_list)

        # tuple_class = ('class', arg)
        tuple_class = max(args_class.items(), key=lambda x: x[1])
        return self.classification[tuple_class[0]]

    def accuracy_rate(self, total):
        corrects = filter(lambda x: int(x[0] == int(x[1])), total)

        return len(list(corrects)) / len(total)

    def solve(self, k):
        result = []
        for v_test in self.data_test:
            k_nearest = self.get_k_nearest(v_test, k)
            arg_max = self.argmax(k_nearest, v_test)
            tup = (arg_max, v_test[self.class_position])
            result.append(tup)

        return self.accuracy_rate(result)