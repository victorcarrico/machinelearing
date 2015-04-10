# coding: utf-8

import numpy as np

from itertools import groupby


class KNN():

    def __init__(self, fname_data, fname_test, class_position):
        """
            class_position indicates position of class argument
            in dataset
        """

        self.raw_data = self._get_raw_data(fname_data)
        self.class_position = class_position
        self.classification = self._find_classes()
        self.data = self._load_data(fname_data)
        self.data_test = self._load_data(fname_test)

    def _get_raw_data(self, fname):
        rdata = []
        with open(fname) as f:
            rdata = [line.split(',') for line in f]
        return rdata

    def _find_classes(self):
        classes = []
        pos = self.class_position
        for vector in self.raw_data:
            if vector[pos] not in classes:
                classes.append(str(vector[pos].strip()))

        classification = dict(zip(classes, range(0, len(classes))))

        return classification

    def _load_data(self, fname):
        with open(fname) as f:
            return np.loadtxt(
                f, delimiter=',', converters={
                self.class_position: lambda x: self.classification[x.decode()]})

    def _delta(self, a, b):
        return 1 if a == b else b

    def _most_commom(self, l):
        l = sorted(l)
        l = [list(group) for key, group in groupby(l)]
        frequency = [(len(i), i[0]) for i in l]

        return max(frequency, key=lambda x:x[0])[1]


    def get_k_nearest(self, vector, k):
        distances = [
            (train_vector, np.linalg.norm(train_vector - vector))
            for train_vector in self.data
        ]
        k_nearest = sorted(distances, key=lambda x: x[1])[:k]

        return k_nearest

    def accuracy_rate(self, result):
        test = [vector[self.class_position] for vector in self.data_test]
        matches = len([i for i, j in zip(result, test) if i == j])
        total = len(self.data_test)
        rate = float(matches/total)

        return rate

    def solve(self, k):
        k_nearest = []
        for item in self.data_test:
            k_nearest.append(self.get_k_nearest(item, k))

        result = []
        for tuples in k_nearest:
            classes = []
            for vector, distance in tuples:
                classes.append(vector[self.class_position])
            result.append(self._most_commom(classes))

        rate = self.accuracy_rate(result)
        print(rate)
        return result