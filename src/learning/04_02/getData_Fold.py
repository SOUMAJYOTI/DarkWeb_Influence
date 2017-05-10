import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold



def getFolds(Y):
    train_folds = []
    test_folds = []
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(Y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        # X_train, X_test = train_index, test_index
        # # Y_train, Y_test = train_index, test_index
        train_folds.append(train_index)
        test_folds.append(test_index)

    return train_folds, test_folds


if __name__ == "__main__":
    forumsData = pd.read_csv('../../../darkweb_data/3_25/Forum40_labels.csv', encoding="ISO-8859-1")
    forumsData = forumsData.fillna(value=0)
    Y_labels = np.array(forumsData.ix[:, 3:14])
    for idx in range(Y_labels.shape[0]):
        for idx_1 in range(Y_labels.shape[1]):
            if Y_labels[idx, idx_1] == 0.:
                Y_labels[idx, idx_1] = -1.
    Y_test_all = np.array([])

    # prepare the folds for CV test
    train_fold, test_fold = getFolds(Y_labels)

    dir_feat = '../../../darkweb_data/3_25/features_d500/'
    for col in range(3, 12):
        X_inst, Y_inst = pickle.load(open(dir_feat + 'feat_label_' + str(col) + '.pickle', 'rb'))
        for idx_y in range(len(Y_inst)):
            if Y_inst[idx_y] == 0:
                Y_inst[idx_y] = -1.
            else:
                Y_inst[idx_y] = 1.

        X_inst = np.array(X_inst)
        X_inst = np.squeeze(X_inst, axis=(1, ))
        Y_inst = np.array(Y_inst)

        cnt_fold = 0
        # print(X_inst[0])
        for idx_t in range(len(train_fold)):
            # X_train = X_inst[:380, :]
            # Y_train = Y_inst[:380]
            # # print(Y_train)
            # X_test = X_inst[380:, :]
            # Y_test = Y_inst[380:]
            # Y_labels_train = Y_labels[:380]

            X_train = X_inst[train_fold[idx_t]]
            Y_train = Y_inst[train_fold[idx_t]]
            Y_labels_train = Y_labels[train_fold[idx_t]]
            X_test = X_inst[test_fold[idx_t]]
            Y_test = Y_inst[test_fold[idx_t]]

            # print(X_train.shape)

            data_fold = [X_train, Y_train, Y_labels_train, X_test, Y_test]
            pickle.dump(data_fold, open('../../../darkweb_data/4_4/folds_data/d500/label_' +
                                        str(col) + '_fold_' + str(idx_t) + '.pickle', 'wb'))