# coding: utf-8

import sys
from itertools import groupby

import matplotlib.pyplot as plt

from vdm import VDM


class KNN_VDM(VDM):

    def _get_distance(self, v_1, v_2):
        return self.vdm(v_1, v_2)

    def get_k_nearest(self, vector, k):
        distances = [
            (train_vector, self._get_distance(train_vector, vector))
            for train_vector in self.training_data
        ]
        k_nearest = sorted(distances, key=lambda x: x[1])[:k]

        return k_nearest

    def get_k_nearest_test(self, k):
        k_nearest = []
        for item in self.testing_data:
            k_nearest.append(self.get_k_nearest(item, k))

        return k_nearest

    def _most_commom(self, l):
        l = sorted(l)
        l = [list(group) for key, group in groupby(l)]
        frequency = [(len(i), i[0]) for i in l]

        return max(frequency, key=lambda x:x[0])[1]

    def accuracy_rate(self, result):
        test = [vector[self.class_position] for vector in self.testing_data]
        matches = len([i for i, j in zip(result, test) if i == j])
        total = len(self.testing_data)
        rate = float(matches/total)

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


def print_result(knn_vdm):
    ks = [1,2,3,5,7,9,11,13,15]
    results = []

    for k in ks:
        rate = knn_vdm.solve(k)
        results.append(rate)
        print("""
            ***k-NN (without weight)***
            k={}
            Accuracy Rate: {}
            """.format(k, rate))

    plt.plot(ks, results)
    plt.show()


def main(argv):

    # knn_vdm = KNN_VDM('hayes-roth.data', 'hayes-roth.test', 5, 0)
    # print('**** Start k-NNN Distance: VDM ****')
    # print_result(knn_vdm)
    # print('****End****')

    knn_vdm = KNN_VDM('yellow-small.data', 'yellow-small.test', 4)
    print('**** Start k-NNN Distance: VDM ****')
    print_result(knn_vdm)
    print('****End****')

if __name__ == '__main__':
    main(sys.argv[1:])