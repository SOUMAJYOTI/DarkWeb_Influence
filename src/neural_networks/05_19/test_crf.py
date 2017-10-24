import itertools
import pickle
import numpy as np
from scipy import sparse

from sklearn.metrics import hamming_loss
from sklearn.datasets import fetch_mldata
from sklearn.metrics import mutual_info_score
from scipy.sparse.csgraph import minimum_spanning_tree

from pystruct.learners import OneSlackSSVM
from pystruct.models import MultiLabelClf
from pystruct.datasets import load_scene
import sklearn.metrics

def chow_liu_tree(y_):
    # compute mutual information using sklearn
    n_labels = y_.shape[1]
    mi = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            mi[i, j] = mutual_info_score(y_[:, i], y_[:, j])
    mst = minimum_spanning_tree(sparse.csr_matrix(-mi))
    edges = np.vstack(mst.nonzero()).T
    edges.sort(axis=1)
    return edges

cnt_fold = 0
for idx_fold in range(0, 1):
    print('\n Fold: ', idx_fold)
    cnt_fold += 1
    X_test = pickle.load(open('../../../darkweb_data/05/5_19/data_test/v3/fold_' + str(idx_fold) +
                              '/' + 'X_test.pickle', 'rb'))
    Y_test_all = pickle.load(open('../../../darkweb_data/05/5_19/data_test/v3/fold_' + str(idx_fold) +
                                  '/' + 'Y_test_all.pickle', 'rb'))
    Y_test_initial = []
    X_test_new = []
    Y_test_new = []

    """ Initial labels from prediction """
    Y_initial = np.zeros(Y_test_all.shape)
    for col in range(2, 12):
        input_dir = '../../../darkweb_data/05/5_19/data_test/v3/fold_' + str(idx_fold) + '/col_' + str(col) + '/'
        X_train = pickle.load(open(input_dir + 'X_train_l.pickle', 'rb'))
        Y_train = pickle.load(open(input_dir + 'Y_train_l.pickle', 'rb'))
        Y_train_all = pickle.load(open(input_dir + 'Y_train_all.pickle', 'rb'))

        Y_train_all = Y_train_all.astype(int)

        independent_model = MultiLabelClf(inference_method='unary')
        independent_ssvm = OneSlackSSVM(independent_model, C=.1, tol=0.01)

        print("fitting independent model...")
        independent_ssvm.fit(X_train, Y_train_all)
        Y_pred = np.array(independent_ssvm.predict(X_test))

        Y_initial[:, col-2] = Y_pred[:, col-2]

    for idx in range(Y_test_all.shape[0]):
        for idx1 in range(Y_test_all.shape[1]):
            if Y_test_all[idx, idx1] == -1.:
                Y_test_all[idx, idx1] = 0.

    Y_test_all = Y_test_all.astype(int)
    # print(Y_initial[:10])
    for col in range(2, 12):
        print(sklearn.metrics.f1_score(Y_test_all[:, col-2], Y_initial[:, col-2]))
