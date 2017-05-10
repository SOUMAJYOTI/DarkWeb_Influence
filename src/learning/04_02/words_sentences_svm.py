import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import AffinityPropagation
import scipy.stats as scst
from sklearn.model_selection import KFold


# Enumerate the sentences into individual instances and then perform an SVM

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


def get_X_Y_data(forumsData, docs, w2v_feat, column):
    dict_doc_sent = {}  # hashmap for documnet sentence map
    positive_posts = []
    negative_posts = []
    positive_indices = []
    negative_indices = []
    positive_row_indices = [] # keep track of +ve instance number for enumerated sentence instances
    negative_row_indices = []  # keep track of -ve instance number for enumerated sentence instances
    row_instances = [] # rows of the filtered sentences
    for row in range(len(forumsData)):
        # print(row)
        if len(docs[row]) == 0:
            v = [list(np.zeros((100,1)))]
        else:
            v = getFeatVectors(docs[row], w2v_feat)
        temp_idx = []
        row_indices = []
        for idx in range(len(v)):
            temp_idx.append((row, idx))
            row_indices.append(row)
            dict_doc_sent[(row, idx)] = v[idx]

        # separate the posts into positive and negative posts
        if forumsData.ix[row, column] == 1.0:
            positive_posts.extend(v)
            positive_indices.extend(temp_idx)
            positive_row_indices.extend(row_indices)
        else:
            negative_posts.extend(v)
            negative_indices.extend(temp_idx)
            negative_row_indices.extend(row_indices)

    X_inst = []
    Y_inst = []

    pos_tags = clusterDoc(positive_posts, pref=-10)

    """ Positive instances """
    for idx in range(len(pos_tags)):
        # print(positive_indices)
        X_inst.append(positive_posts[pos_tags[idx]])
        row_instances.append(positive_row_indices[pos_tags[idx]])
        # X_inst.append(dict_doc_sent[positive_indices[pos_tags[idx]]])
        Y_inst.append(1.)

    for idx in range(len(positive_row_indices)):
        if positive_row_indices[idx] not in row_instances:
            for pos_idx in range(10):
                try:
                    X_inst.append(dict_doc_sent[(positive_row_indices[idx], pos_idx)])
                    row_instances.append(positive_row_indices[idx])
                    Y_inst.append(1.)
                except:
                    break

    """ Negative instances """
    neg_tags = clusterDoc(negative_posts, pref=-10)
    for idx in range(len(neg_tags)):
        X_inst.append(negative_posts[neg_tags[idx]])
        row_instances.append(negative_row_indices[neg_tags[idx]])
        # X_inst.append(dict_doc_sent[negative_indices[neg_tags[idx]]])
        Y_inst.append(-1.)

    for idx in range(len(negative_row_indices)):
        if negative_row_indices[idx] not in row_instances:
            row_instances.append(negative_row_indices[idx])
            X_inst.append(dict_doc_sent[(negative_row_indices[idx], 0)])
            Y_inst.append(-1.)

    # print(count_neg)
    return X_inst, Y_inst, row_instances, len(pos_tags) / len(positive_indices), len(neg_tags) / len(negative_indices)



# if __name__ == "__main__":
    # output_dir = '../../../darkweb_data/3_20/'
    # dict_phrases_file = output_dir + 'dict_sent_phrases_sg_3.txt'
    # dict_phrases = open(dict_phrases_file, 'r')
    #
    # inputData = open(output_dir + 'partitionOut_sg_3.txt', 'r')
    # stopwords_file = open('../../../darkweb_data/3_25/Stop_Words.txt', 'r')
    #
    # # Load the trained word2vec model and the sentences
    # sentences_data = open('../../../darkweb_data/3_20/forum_40_input_phrases_indexed.txt', 'r')
    # # docs = createSentences(sentences_data)
    #
    # dictionary = getDictPhrases(dict_phrases)
    # stopwords = getStopWords(stopwords_file)
    #
    # docs = createSentences(sentences_data, inputData, dictionary, stopwords)
    # pickle.dump(docs, open('../../../darkweb_data/3_25/docs_corpora.pickle', 'wb'))
    #
    # w2v_feat = pickle.load(open('../../../darkweb_data/3_25/word2vec_train_model_d100.pickle', 'rb'))
    # # d2v_feat = gensim.models.Doc2Vec.load('../../../darkweb_data/3_25/doc2vec_train_model_d50.model')
    #
    # # For each label, create the clusters of positive and negative sentences
    # forumsData = pd.read_csv('../../../darkweb_data/3_25/Forum40_labels.csv', encoding="ISO-8859-1")
    # forumsData = forumsData.fillna(value=0)
    #
    # # dir_save = '../../../darkweb_data/3_25/features_d500/'
    # col_names = list(forumsData.columns.values)
    # Y_test_all = np.array([])
    #
    # Y_labels = np.array(forumsData.ix[:, 3:14])
    # for idx in range(Y_labels.shape[0]):
    #     for idx_1 in range(Y_labels.shape[1]):
    #         if Y_labels[idx, idx_1] == 0.:
    #             Y_labels[idx, idx_1] = -1.
    #
    # # SVM classifier
    # clf = svm.SVC(kernel='rbf', C=1000)
    # # clf = linear_model.LogisticRegression()
    #
    # for col in range(3, 12):
    #     X_inst, Y_inst, pos_perc, neg_perc = get_X_Y_data(forumsData, docs, w2v_feat, col)
    #     # pickle.dump((X_inst, Y_inst), open(dir_save + 'feat_label_' + str(col) + '.pickle', 'wb'))
    #
    #     # prepare the folds for CV test
    #     train_fold, test_fold = getFolds(Y_inst)
    #     X_inst = np.array(X_inst)
    #     # X_inst = np.squeeze(X_inst, axis=(1,))
    #     Y_inst = np.array(Y_inst)
    #
    #     avg_precision = 0
    #     avg_recall = 0
    #     avg_f1 = 0
    #
    #     avg_precision_r = 0
    #     avg_recall_r = 0
    #     avg_f1_r = 0
    #
    #     perc_pos = 0
    #
    #     # print(X_inst[0])
    #     for idx_t in range(len(train_fold)):
    #         # print(len(train_fold[idx_t]))
    #         X_train = X_inst[train_fold[idx_t]]
    #         Y_train = Y_inst[train_fold[idx_t]]
    #         Y_labels_train = Y_labels[train_fold[idx_t]]
    #         X_test = X_inst[test_fold[idx_t]]
    #         Y_test = Y_inst[test_fold[idx_t]]
    #
    #         # print((len(Y_test[Y_test == 1.]) + len(Y_train[Y_train == 1.]) )/ (len(Y_test) + len(Y_train)))
    #         # perc_pos += len(Y_test[Y_test == 1.]) / len(Y_test)
    #         clf.fit(X_train, Y_train)
    #
    #         Y_random = []
    #         for idx_r in range(X_test.shape[0]):
    #             Y_random.append(random.sample(range(2), 1))
    #
    #         Y_random = np.array(Y_random)
    #
    #         Y_predict = clf.predict(X_test)
    #         avg_precision += sklearn.metrics.f1_score(Y_test, Y_predict)
    #
    #     print(avg_precision/10)
    #
    #
