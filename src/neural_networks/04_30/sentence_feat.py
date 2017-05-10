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
def getFeatVectors(doc, w2v):
    doc_vec = []
    doc_mean_vec = []
    for d in range(len(doc)):
        sent_vec = []
        phrases = doc[d]
        phr_sorted = sorted(phrases.items(), key=operator.itemgetter(1), reverse=True)
        if len(phr_sorted) > 30:
            phr_sorted = phr_sorted[:30]

        for item in range(len(phr_sorted)):
            p, tv = phr_sorted[item]
            words = p.split('_')
            if len(words) >= 2:
                for w in words:
                    # if w in stopwords:
                    #     continue
                    if w in w2v:
                        sent_vec.append(w2v[w])

            else:
                if p in w2v:
                    sent_vec.append(w2v[p])

        if len(sent_vec) == 0:
            continue

        # Cases where the number of words is less than 30 for a sentence
        if len(sent_vec) > 30:
            sent_vec = sent_vec[:30]
        else:
            l = len(sent_vec)
            for idx_fill in range(30-l):
                sent_vec.append(np.random.uniform(-1, 1, (50))) # pad with random distribution

        doc_mean_vec.append(np.mean(sent_vec, axis=0))
        doc_vec.append(sent_vec)

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
def get_X_Y_data(forumsData, docs, docs_unlabel, w2v_feat, column):
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
            act_v = list(np.zeros((1, 30, 50)))
            v = [list(np.random.uniform(-1, 1, (50)))]
        else:
            act_v, v = getFeatVectors(docs[row], w2v_feat)
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
            act_v = list(np.zeros((1, 30, 50)))
            v = [list(np.random.uniform(-1, 1, (50)))]
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
    # pos_tags = clusterDoc(positive_posts, pref=-30)
    cnt_pos = 0
    # for idx in range(len(positive_row_indices)):
    #     # print(positive_indices)
    #     # print(positive_posts[pos_tags[idx]])
    #     # X_inst.append(positive_posts[pos_tags[idx]])
    #     row_instances.append(positive_row_indices[pos_tags[idx]])
    #     X_inst_2D.append(dict_doc_sent[positive_indices[pos_tags[idx]]])
    #     X_inst_3D.append(dict_doc_sent_3D[positive_indices[pos_tags[idx]]])
    #     Y_inst.append(1.)
    #     cnt_pos += 1

    for idx in range(len(positive_row_indices)):
        if positive_row_indices[idx] not in row_instances:
            for pos_idx in range(50):
                try:
                    X_inst_2D.append(dict_doc_sent[(positive_row_indices[idx], pos_idx)])
                    X_inst_3D.append(dict_doc_sent_3D[(positive_row_indices[idx], pos_idx)])
                    row_instances.append(positive_row_indices[idx])
                    Y_inst.append(1.)
                    cnt_pos += 1
                except:
                    break

    # print(np.array(X_inst_2D).shape)
    """ Negative instances """
    neg_tags = []
    # neg_tags = clusterDoc(negative_posts, pref=-10)
    # for idx in range(len(neg_tags)):
    #     X_inst.append(negative_posts[neg_tags[idx]])
    #     row_instances.append(negative_row_indices[neg_tags[idx]])
    #     # X_inst.append(dict_doc_sent[negative_indices[neg_tags[idx]]])
    #     Y_inst.append(-1.)

    """ NUMBER OF POSITIVE TRAINING EXAMPLES = NUMBER OF NEGATIVE TRAINING EXAMPLES
        FOR TRANSDUCTIVE TRAINING
     """
    cnt_neg = 0
    for idx in range(len(negative_row_indices)):
        if negative_row_indices[idx] not in row_instances:
            for neg_idx in range(50):
                try:
                    X_inst_2D.append(dict_doc_sent[(negative_row_indices[idx], neg_idx)])
                    X_inst_3D.append(dict_doc_sent_3D[(negative_row_indices[idx], neg_idx)])
                    row_instances.append(negative_row_indices[idx])
                    Y_inst.append(-1.)
                    cnt_neg += 1
                except:
                    break

    # print(np.array(X_inst_2D).shape)
            # if cnt_pos < cnt_neg:
            #     break

    # print(cnt_pos, cnt_neg)

    # """ Unlabeled Instances """
    # # print(unlabel_posts[:10])
    # # unlabel_tags = clusterDoc(unlabel_posts, pref)
    #
    for idx in range(2000):
        X_inst_2D.append(unlabel_posts[idx])
        X_inst_3D.append(unlabel_posts_3D[idx])
        X_unlabel.append(unlabel_posts_3D[idx])
        # row_instances.append(unlabel_row_indices[unlabel_tags[idx]])
        # X_inst.append(dict_doc_sent[positive_indices[pos_tags[idx]]])
        Y_inst.append(2.)

    # print(Y_inst)
    return X_inst_2D,  X_inst_3D, X_unlabel, Y_inst, row_instances

