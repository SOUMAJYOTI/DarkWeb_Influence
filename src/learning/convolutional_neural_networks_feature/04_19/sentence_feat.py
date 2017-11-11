import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import AffinityPropagation
import scipy.stats as scst
from sklearn.model_selection import KFold
import operator
from sklearn.decomposition import PCA

""" Enumerate the sentences into individual instances to get s X d matrix """
""" Sentence features - stack top 10 words by tfidf into a matrix for that sentence"""

def getDictPhrases(dict_phrases):
    phrasesWords = {}
    for line in dict_phrases:
        w = line.split(',')
        phrasesWords[w[0]] = int(w[2][:len(w[2]) - 1])

    return phrasesWords


def getStopWords(data):
    for line in data:
        words = line.split(' ')
    # print(len(words))
    return words


# it gets the w2v for the phrases as well as words
# the sentence feature is the average of the word word2vec
def getFeatVectors(doc, w2v_feat):
    doc_vec = []
    doc_mean_vec = []
    for d in range(len(doc)):
        sent_vec = []
        phrases = doc[d]
        phr_sorted = sorted(phrases.items(), key=operator.itemgetter(1))
        if len(phr_sorted) > 10:
            phr_sorted = phr_sorted[:10]

        for item in range(len(phr_sorted)):
            p, tv = phr_sorted[item]
            words = p.split('_')
            if len(words) >= 2:
                temp_sum = 0.
                count_w = 0
                for w in words:
                    if w in w2v_feat:
                        count_w += 1
                        temp_sum += w2v_feat[w]
                temp_sum = temp_sum/count_w
                sent_vec.append(temp_sum)
            else:
                if p in w2v_feat:
                    sent_vec.append(w2v_feat[p])

        if len(sent_vec) == 0:
            continue
        # print(np.array(sent_vec).shape)
        sent_vec = np.pad(sent_vec, ((0, 10 - len(sent_vec)), (0,0)), 'constant', constant_values=0)

        # print(np.array(sent_vec))
        doc_mean_vec.append(np.mean(np.array(sent_vec), axis=0))
        doc_vec.append(np.array(sent_vec))

    # print(np.array(doc_mean_vec).shape)
    return np.array(doc_vec), np.array(doc_mean_vec)


def clusterDoc(featDocs, pref):

    # Compute Affinity Propagation
    # TODO - Try other clustering algos if time permits !!!
    af = AffinityPropagation(preference=pref).fit(featDocs)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    # max_occur = max(set(labels), key=labels.count)
    max_label = scst.mode(labels, axis=None)[0][0]

    tagged_indices = []
    for l in range(len(labels)):
        if labels[l] == max_label:
            tagged_indices.append(l)
    return tagged_indices


# Schema for instances for recognition
# 1. - positive instance
# -1. - negative instance
# 2. - unlabeled instance
def get_X_Y_data(forumsData, docs, docs_unlabel, w2v_feat, column, trans_data):
    dict_doc_sent = {}  # hashmap for documnet sentence map
    dict_doc_sent_3D = {}
    dict_doc_sent_unlabel = {} # hashmap for unlabeled documnet sentence map

    positive_posts = []
    negative_posts = []
    unlabel_posts = []
    unlabel_posts_3D = []

    positive_indices = []
    negative_indices = []
    unlabel_indices = []

    positive_row_indices = [] # keep track of +ve instance number for enumerated sentence instances
    negative_row_indices = []  # keep track of -ve instance number for enumerated sentence instances
    unlabel_row_indices = [] #  # keep track of unlabel instance number for enumerated sentence instances
    row_instances = [] # rows of the filtered sentences
    for row in range(len(forumsData)):
        if len(docs[row]) == 0:
            act_v = list(np.zeros((1, 10, 200)))
            v = [list(np.zeros((200, 1)))]
        else:
            act_v , v = getFeatVectors(docs[row], w2v_feat)
            # print(np.array(act_v).shape, len(v))
        temp_idx = []
        row_indices = []
        for idx in range(len(v)):
            temp_idx.append((row, idx))
            row_indices.append(row)
            # print(np.array(act_v[idx]).shape)
            dict_doc_sent[(row, idx)] = v[idx]
            dict_doc_sent_3D[(row, idx)] = act_v[idx]

        # separate the posts into positive and negative posts
        if forumsData.ix[row, column] == 1.0:
            positive_posts.extend(v)
            positive_indices.extend(temp_idx)
            positive_row_indices.extend(row_indices)
        else:
            negative_posts.extend(v)
            negative_indices.extend(temp_idx)
            negative_row_indices.extend(row_indices)

    # print(np.array(negative_posts).shape)
    # Append the unlabeled posts
    for row in range(len(docs_unlabel)):
        # print(row)
        if len(docs_unlabel[row]) == 0:
            act_v = list(np.zeros((1, 10, 200)))
            v = [list(np.zeros((200, 1)))]
        else:
            act_v, v = getFeatVectors(docs_unlabel[row], w2v_feat)
        temp_idx = []
        row_indices = []
        for idx in range(len(v)):
            temp_idx.append((row, idx))
            row_indices.append(row)

        unlabel_posts.extend(v)
        unlabel_posts_3D.extend(act_v)
        unlabel_indices.extend(temp_idx)
        unlabel_row_indices.extend(row_indices)

    X_inst_2D = []
    X_inst_3D = []
    X_unlabel = []
    Y_inst = []

    """ Positive instances """
    pos_tags = clusterDoc(positive_posts, pref=-30)
    for idx in range(len(pos_tags)):
        # print(positive_indices)
        # print(positive_posts[pos_tags[idx]])
        # X_inst.append(positive_posts[pos_tags[idx]])
        row_instances.append(positive_row_indices[pos_tags[idx]])
        X_inst_2D.append(dict_doc_sent[positive_indices[pos_tags[idx]]])
        X_inst_3D.append(dict_doc_sent_3D[positive_indices[pos_tags[idx]]])
        Y_inst.append([1.])

    for idx in range(len(positive_row_indices)):
        if positive_row_indices[idx] not in row_instances:
            for pos_idx in range(20):
                try:
                    X_inst_2D.append(dict_doc_sent[(positive_row_indices[idx], pos_idx)])
                    X_inst_3D.append(dict_doc_sent_3D[(positive_row_indices[idx], pos_idx)])
                    row_instances.append(positive_row_indices[idx])
                    Y_inst.append([1.])
                except:
                    break

    """ Negative instances """
    neg_tags = []
    # neg_tags = clusterDoc(negative_posts, pref=-10)
    # for idx in range(len(neg_tags)):
    #     X_inst.append(negative_posts[neg_tags[idx]])
    #     row_instances.append(negative_row_indices[neg_tags[idx]])
    #     # X_inst.append(dict_doc_sent[negative_indices[neg_tags[idx]]])
    #     Y_inst.append(-1.)

    for idx in range(len(negative_row_indices)):
        if negative_row_indices[idx] not in row_instances:
            row_instances.append(negative_row_indices[idx])
            X_inst_2D.append(dict_doc_sent[(negative_row_indices[idx], 0)])
            X_inst_3D.append(dict_doc_sent_3D[(negative_row_indices[idx], 0)])
            Y_inst.append([-1.])

    """ Unlabeled Instances """
    # print(unlabel_posts[:10])
    # unlabel_tags = clusterDoc(unlabel_posts, pref)

    for idx in range(0): #range(len(unlabel_posts)):
        X_inst_2D.append(unlabel_posts[idx])
        X_inst_3D.append(unlabel_posts_3D[idx])
        X_unlabel.append(unlabel_posts_3D[idx])
        # row_instances.append(unlabel_row_indices[unlabel_tags[idx]])
        # X_inst.append(dict_doc_sent[positive_indices[pos_tags[idx]]])
        Y_inst.append([2.])

    return X_inst_2D,  X_inst_3D, X_unlabel, Y_inst, row_instances

