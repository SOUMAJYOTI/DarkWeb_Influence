import pandas as pd
import numpy as np
import pickle
import cvxopt
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import scipy.stats as scst
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import gensim
from imblearn.over_sampling import SMOTE
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import KFold
import sklearn.metrics
from sklearn.naive_bayes import GaussianNB
import random
import matplotlib.pyplot as plt
import seaborn as sns


def hamming_loss(Y_true, Y_predict):
    hamming_sum = 0
    for inst in range(Y_true.shape[0]):
        hamming_sum += sklearn.metrics.hamming_loss(Y_true[inst,:], Y_predict[inst, :])

    return hamming_sum/Y_true.shape[0]


def accuracy(Y_true, Y_predict):
    jaccard_sum = 0
    for inst in range(Y_true.shape[0]):
        jaccard_sum += sklearn.metrics.jaccard_similarity_score(Y_true[inst,:], Y_predict[inst, :])

    return jaccard_sum/Y_true.shape[0]


def exact_match(Y_true, Y_predict):
    exact_sum = 0
    for inst in range(Y_true.shape[0]):
        if Y_true[inst,:] == Y_predict[inst, :]:
            exact_sum += 1

    return exact_sum / Y_true.shape[0]


def F1_measure(Y_true, Y_predict):
    f1_sum = 0
    for inst in range(Y_true.shape[0]):
        f1_sum += sklearn.metrics.f1_score(Y_true[inst, :], Y_predict[inst, :])

    return f1_sum / Y_true.shape[0]


def macro_f1(Y_true, Y_predict):
    f1_sum = 0
    for l in range(Y_true.shape[1]):
        f1_sum += sklearn.metrics.f1_score(Y_true[:, l], Y_predict[:, l])

    return f1_sum / Y_true.shape[1]


def micro_f1(Y_true, Y_predict):
    numer = 0
    for l in range(Y_true.shape[1]):
        for inst in range(Y_true.shape[0]):
            numer += (Y_true[inst, l] * Y_predict[inst, l])

    numer *= 2

    denom_1 = 0
    denom_2 = 0
    for l in range(Y_true.shape[1]):
        for inst in range(Y_true.shape[0]):
            denom_1 += Y_true[inst, l]

    for l in range(Y_true.shape[1]):
        for inst in range(Y_true.shape[0]):
            denom_2 += (Y_predict[inst, l])

    denom = denom_1 + denom_2

    return numer/denom


# def average_precision(Y_true, Y_predict):


