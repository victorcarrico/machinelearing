# coding: utf-8

import sys

import matplotlib.pyplot as plt

from knn import KNNWeighted


def print_result(knn, training_set, testing_set):
    ks = [1,2,3,5,7,9,11,13,15]
    results = []

    for k in ks:
        rate = knn.solve(k)
        results.append(rate)
        print("""
            ***k-NN (with weight)***
            k={}
            Training set lenght: {}
            Testing set lenght: {}
            Accuracy Rate: {}
            """.format(k, len(training_set), len(testing_set), rate))

    plt.plot(ks, results)
    plt.show()


def main(argv):

    knnw = KNNWeighted('iris.data', 'iris.test', 4)
    training_set = knnw.get_training_set()
    testing_set = knnw.get_testing_set()
    print('**********\n***** IRIS ****\n***********')
    print_result(knnw, training_set, testing_set)

    knnw = KNNWeighted('wdbc.data', 'wdbc.test', 1, 0)
    print('**********\n***** Breast Cancer in Wisconsin ****\n***********')
    training_set = knnw.get_training_set()
    testing_set = knnw.get_testing_set()
    print_result(knnw, training_set, testing_set)

if __name__ == '__main__':
    main(sys.argv[:1])