from knn import KNN


def main():

    KNearest = KNN('wdbc.data', 'wdbc.test', 1)
    # KNearest.get_k_nearest(KNearest.data_test[1], 5)
    KNearest.solve(5)
        

    # print('Data {}: '.format(data))
    # print('Data Test: {}'.format(data_test))


if __name__ == '__main__':
    main()