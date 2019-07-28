import pandas as pd
import numpy as np
from sklearn.utils import shuffle

class GQ_MS:

    class Data:

        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, file):

        trn, val, tst = load_data_and_clean_and_split(file)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]


def load_data(file):

    # data = pd.read_pickle(file)
    # data = pd.read_pickle(file).sample(frac=0.25)
    # data.to_pickle(file)
    data = pd.read_excel(file)
    data = shuffle(data)
    data.reset_index(inplace=True, drop=True)
    return (data - data.mean()) / data.std()

#
# def get_correlation_numbers(data):
#     C = data.corr()
#     A = C > 0.98
#     B = A.values.sum(axis=1)
#     return B
#
#
# def load_data_and_clean(file):
#
#     data = load_data(file)
#     B = get_correlation_numbers(data)
#
#     while np.any(B > 1):
#         col_to_remove = np.where(B > 1)[0][0]
#         col_name = data.columns[col_to_remove]
#         data.drop(col_name, axis=1, inplace=True)
#         B = get_correlation_numbers(data)
#     # print(data.corr())
#     data = (data - data.mean()) / data.std()
#
#     return data


def load_data_and_clean_and_split(file):

    data = load_data(file).values
    N_test = int(0.25 * data.shape[0])
    data_test = data[-N_test:]
    data_train = data[0:-N_test]
    N_validate = int(0.25 * data_train.shape[0])
    data_validate = data_train[-N_validate:]
    # data_train = data_train[0:-N_validate]

    return data_train, data_test, data_test
