# coding: utf-8

import sys

from knn import KNN


def print_result(knn, training_set, testing_set):
    ks = [1,2,3,5,7,9,11,13,15]

    for k in ks:
        rate = knn.solve(k)
        print("""
            ***k-NN (with weight)***
            k={}
            Training set lenght: {}
            Testing set lenght: {}
            Accuracy Rate: {}
            """.format(k, len(training_set), len(testing_set), rate))


def main(argv):


    knn = KNN('iris.data', 4)
    training_set = knn.get_training_set()
    testing_set = knn.get_testing_set()
    print('**********\n***** IRIS ****\n***********')
    print_result(knn, training_set, testing_set)

    knn = KNN('wdbc.data', 1, 0)
    print('**********\n***** Breast Cancer in Wisconsin ****\n***********')
    training_set = knn.get_training_set()
    testing_set = knn.get_testing_set()
    print_result(knn, training_set, testing_set)

if __name__ == '__main__':
    main(sys.argv[1:])