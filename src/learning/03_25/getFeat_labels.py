import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import scipy.stats as scst
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import gensim
from imblearn.over_sampling import SMOTE
from sklearn import linear_model


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


def createSentences(data, phrase_sentences, dict, stopwords):
    docs_phrases = []
    for line in phrase_sentences:
        line = line[:len(line)-1]
        doc_phr = line.split(',')
        line_phrase = []
        for ph in range(len(doc_phr)):
            phrases = doc_phr[ph]
            p = phrases.split(':')
            if p[0] not in dict:
                continue
            count = dict[p[0]]
            words = p[0].split(' ')

            # separate conditions for single words and sentences !!!
            # TODO - ADD TF_IDF OR TERM WEIGHTED FILTER !!!!
            #1. Single word filter - remove stopwords
            if len(words) < 2:
                # print(words, count)
                if p[0] in stopwords:
                    continue
                # print(words, count)
                if count > 10 and count < 200:
                    line_phrase.append(p[0])
            #2. Phrase filter - no filter except for all stopwords
            else:
                merged_phrase = ''
                count_stops = 0
                for w in range(len(words)):
                    if w in stopwords:
                        count_stops += 1
                    merged_phrase += (words[w] + ' ')
                if count_stops == len(words):
                    continue
                line_phrase.append(merged_phrase[:len(merged_phrase)-1])

            # print(line_phrase)
        docs_phrases.append(line_phrase)

    # print(docs_phrases[len(docs_phrases)-2:])
    # merge the separate sentences into one doc based on the original file
    sent_list = []
    docs_list = []
    cur_doc = 0
    count_line = 0
    for line in data:
        words = line.split(' ')
        sent_index = int(words[0])
        if sent_index != cur_doc:
            cur_doc += 1
            docs_list.append(sent_list)
            sent_list = []

        if len(docs_phrases[count_line]) > 0:
            sent_list.append(docs_phrases[count_line])
        count_line += 1

    docs_list.append(sent_list)
    # print(docs_list[41])
    return docs_list


def getDocFeatVectors(doc, d2v_feat):
    doc_vec = []
    for d in range(len(doc)):
        sent = doc[d]
        sent_vec = d2v_feat.infer_vector(sent, alpha=0.025, min_alpha=0.0001)

        doc_vec.append(sent_vec)

    return np.array(doc_vec)


# it gets the w2v for the phrases as well as words
# the senetnce feature is the average of the word word2vec
def getFeatVectors(doc, w2v_feat):
    doc_vec = []
    for d in range(len(doc)):
        sent = doc[d]
        sum_w2v = []
        for s in sent:
            words = s.split()
            if len(words) >= 2:
                # print(s)
                temp_sum = 0.
                count_w = 0
                for w in words:
                    if w in w2v_feat:
                        count_w += 1
                        temp_sum += w2v_feat[w]
                temp_sum = temp_sum/count_w
                sum_w2v.append(temp_sum)
            else:
                if s in w2v_feat:
                    sum_w2v.append(w2v_feat[s])

        sent_vec = np.mean(np.array(sum_w2v), axis=0)
        doc_vec.append(sent_vec)

    return np.array(doc_vec)


def clusterDoc(featDocs, pref):

    # Compute Affinity Propagation
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


def get_X_Y_data(forumsData, docs, w2v_feat, column):
    dict_doc_sent = {}  # hashmap for documnet sentence map
    positive_posts = []
    negative_posts = []
    positive_indices = []
    negative_indices = []
    for row in range(len(forumsData)):
        # print(row)
        if len(docs[row]) == 0:
            v = [list(np.zeros((50,1)))]
        else:
            v = getFeatVectors(docs[row], w2v_feat)
        temp_idx = []
        for idx in range(len(v)):
            temp_idx.append((row, idx))
            dict_doc_sent[(row, idx)] = v[idx]

        # separate the posts into positive and negative posts
        if forumsData.ix[row, column] == 1.0:
            positive_posts.extend(v)
            positive_indices.extend(temp_idx)
        else:
            negative_posts.extend(v)
            negative_indices.extend(temp_idx)

    # print(len(forumsData))
    X_inst = [[] for _ in range(len(forumsData))]
    Y_inst = [0.0 for _ in range(len(forumsData))]

    pos_tags = clusterDoc(positive_posts, pref=-1)
    print(len(pos_tags) / len(positive_indices))
    sent_doc_idx = []
    for idx in range(len(pos_tags)):
        sent_doc_idx.append(positive_indices[pos_tags[idx]])

    # print(sent_doc_idx)
    # print(sent_doc_idx)

    # pca = PCA(n_components=1)
    positive_feat = []
    count_pos = 0
    for row in range(len(forumsData)):
        if forumsData.ix[row, column] == 1.0:
            count_pos += 1
            v = getFeatVectors(docs[row], w2v_feat)
            row_feat = np.array([])
            for idx in range(len(v)):
                if (row, idx) in sent_doc_idx:
                    # row_feat.append(np.array(v[idx]))
                    row_feat = np.hstack((row_feat, np.array(v[idx])))
                    # take the first 4 sentences as long as there are so many (50 dim)
                    # this way the structure is maintained
                    if row_feat.shape[0] >= 200:
                        break

            if len(row_feat) == 0:
                if len(v) == 0:
                    row_feat = np.zeros((1, 200))
                else:
                    row_feat = v
                row_feat = np.reshape(row_feat, (1, row_feat.shape[0] * row_feat.shape[1]))
            else:
                row_feat = np.reshape(row_feat, (1, row_feat.shape[0]))

            if row_feat.shape[1] < 200:
                while row_feat.shape[1] < 200:
                    # row_feat.append(np.zeros(100))
                    # print(row_feat.shape)
                    row_feat = np.hstack((row_feat, np.zeros((1, 200))))

            row_feat = row_feat[:, :200]
            # row_feat =
            # new_X = pca.fit_transform(row_feat.transpose())
            # positive_feat.append(new_X.transpose())
            X_inst[row] = row_feat #new_X.transpose()
            Y_inst[row] = np.array([1.0])
    # print(np.array(positive_feat).shape)

    neg_tags = clusterDoc(negative_posts, pref=-10)
    print(len(neg_tags) / len(negative_indices))
    sent_doc_idx = []
    for idx in range(len(neg_tags)):
        sent_doc_idx.append(negative_indices[neg_tags[idx]])
    # print(sent_doc_idx)

    pca = PCA(n_components=1)
    negative_feat = []
    count_neg = 0
    for row in range(len(forumsData) ):
        if forumsData.ix[row, column] == 0.0:
            count_neg += 1
            v = getFeatVectors(docs[row], w2v_feat)
            row_feat = np.array([])
            for idx in range(len(v)):
                if (row, idx) in sent_doc_idx:
                    row_feat = np.hstack((row_feat, np.array(v[idx])))
                    if row_feat.shape[0] >= 200:
                        break

            if len(row_feat) == 0:
                if len(v) == 0:
                    row_feat = np.zeros((1, 200))
                else:
                    row_feat = v
                row_feat = np.reshape(row_feat, (1, row_feat.shape[0] * row_feat.shape[1]))
            else:
                row_feat = np.reshape(row_feat, (1, row_feat.shape[0]))
            # row_feat = np.reshape(row_feat, (1, row_feat.shape[0] * row_feat.shape[1]))

            if row_feat.shape[1] < 200:
                while row_feat.shape[1] < 200:
                    # row_feat.append(np.zeros(100))
                    row_feat = np.hstack((row_feat, np.zeros((1, 200))))
            row_feat = row_feat[:,:200]

            # row_feat = row_feat[:5]
            # print(row_feat.shape)
            # new_X = pca.fit_transform(row_feat.transpose())
            # negative_feat.append(new_X.transpose())
            X_inst[row] = row_feat # new_X.transpose()
            Y_inst[row] = np.array([0.0])
            # if count_neg > 3*count_pos:
            #     break
    # print(np.array(negative_feat).shape)

    # print(count_neg)
    return X_inst, Y_inst, len(pos_tags) / len(positive_indices), len(neg_tags) / len(negative_indices)

if __name__ == "__main__":
    output_dir = '../../../darkweb_data/3_20/'
    dict_phrases_file = output_dir + 'dict_sent_phrases_sg_3.txt'
    dict_phrases = open(dict_phrases_file, 'r')

    inputData = open(output_dir + 'partitionOut_sg_3.txt', 'r')
    stopwords_file = open('../../../darkweb_data/3_25/Stop_Words.txt', 'r')

    # Load the trained word2vec model and the sentences
    sentences_data = open('../../../darkweb_data/3_20/forum_40_input_phrases_indexed.txt', 'r')
    # docs = createSentences(sentences_data)

    dictionary = getDictPhrases(dict_phrases)
    stopwords = getStopWords(stopwords_file)

    docs = createSentences(sentences_data, inputData, dictionary, stopwords)
    w2v_feat = pickle.load(open('../../../darkweb_data/3_25/word2vec_train_model_d50.pickle', 'rb'))
    # d2v_feat = gensim.models.Doc2Vec.load('../../../darkweb_data/3_25/doc2vec_train_model_d50.model')

    # For each label, create the clusters of positive and negative sentences
    forumsData = pd.read_csv('../../../darkweb_data/3_25/Forum40_labels.csv', encoding="ISO-8859-1")
    forumsData = forumsData.fillna(value=0)

    dir_save = '../../../darkweb_data/3_25/features_d200/'
    col_names = list(forumsData.columns.values)
    Y_test_all = np.array([])
    for col in range(3, 12):
        # print(forumsData.ix[:, col])
        X_inst, Y_inst, pos_perc, neg_perc = get_X_Y_data(forumsData, docs, w2v_feat, col)
        pickle.dump((X_inst, Y_inst), open(dir_save + 'feat_label_' + str(col) + '.pickle', 'wb'))
