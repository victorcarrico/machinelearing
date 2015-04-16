# coding: utf-8

import numpy as np


class VDM(object):

    def __init__(self, training_data, testing_data, class_position, skip_column=None):
        self.class_position = self._get_class_position(class_position, skip_column)
        self.training_data = self._load_data(training_data, skip_column)
        self.testing_data = self._load_data(testing_data, skip_column)
        self.classes = self._find_classes()
        self.attrs = self._get_attrs()
        self.q = 1

    def _get_class_position(self, class_position, skip_column):
        if skip_column:
            if skip_column < class_position:
                return class_position - 1
        return class_position

    def _get_attrs(self):
        attrs = list(self.training_data[0])
        attrs.pop(self.class_position)

        return len(attrs)

    def _load_data(self, fname, skip_column):
        with open(fname) as f:
            m = np.loadtxt(f, delimiter=',', dtype='str')
            if skip_column:
                m = np.delete(m, skip_column, 1)

            return m

    def _find_classes(self):
        classes = []
        pos = self.class_position
        for vector in self.training_data:
            if vector[pos] not in classes:
                classes.append(str(vector[pos].strip()))

        return classes

    def _get_N_i(self, attr, column):
        column = self.training_data[:, column]

        return list(column).count(attr)

    def _get_N_ic(self, attr, klass, column):
        pos = self.class_position
        f = filter(lambda x: x[pos] == klass, self.training_data)
        column = np.asmatrix(list(f))[:,column]

        return list(column).count(attr)


    def vdm_i(self, vector_a, vector_b, i):
        args = []
        n_ia = self._get_N_i(vector_a[i], i)
        n_ib = self._get_N_i(vector_b[i], i)
        for klass in self.classes:
            n_iac = self._get_N_ic(vector_a[i], klass, i)
            n_ibc = self._get_N_ic(vector_b[i], klass, i)
            try:
                arg = ((n_iac/n_ia) - (n_ibc/n_ib)) ** 2
            except:
                if n_ia == 0:
                    n_ia = 0.0001
                if n_ib == 0:
                    n_ib = 0.0001
                arg = ((n_iac/n_ia) - (n_ibc/n_ib)) ** 2
            args.append(arg)

        return sum(args)

    def vdm(self, vector_a, vector_b):
        args = []
        for i in range(0, self.attrs):
            arg = self.vdm_i(vector_a, vector_b, i)
            args.append(arg)
        vdm = sum(args) ** (0.5)

        return vdm
