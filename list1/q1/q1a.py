from knn import KNN


def main():

    KNearest = KNN('iris.data', 'iris.test')
    # KNearest.get_k_nearest(KNearest.data_test[1], 5)
    KNearest.solve(17)
        

    # print('Data {}: '.format(data))
    # print('Data Test: {}'.format(data_test))


if __name__ == '__main__':
    main()