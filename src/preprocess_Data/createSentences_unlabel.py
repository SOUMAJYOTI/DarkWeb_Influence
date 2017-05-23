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


def createSentences(data, phrase_sentences, dict, stopwords, tf_idf):
    docs_phrases = []
    cnt_line = 0
    for line in phrase_sentences:
        # print(tf_idf[cnt_line])
        # print(line)

        line = line[:len(line)-1]
        doc_phr = line.split(',')
        line_phrase = {}
        for ph in range(len(doc_phr)):
            phrases = doc_phr[ph]
            p = phrases.split(':')

            if p[0] not in dict:
                continue
            words = p[0].split(' ')

            # separate conditions for single words and sentences !!!
            #1. Single word filter - remove stopwords
            if len(words) < 2:
                # print(words, count)
                # TODO: SELECT THRESHOLD FOR TF-IDF
                if p[0] in stopwords:
                    continue
                if p[0] not in tf_idf[cnt_line]:
                    continue
                # if tf_idf[cnt_line][p[0]] < 0.5:
                #     continue

                line_phrase[p[0]] = tf_idf[cnt_line][p[0]]

            #2. Phrase filter - no filter except for all stopwords
            else:
                merged_phrase = ''
                for w in range(len(words)):
                    if words[w] in stopwords:
                        continue
                    merged_phrase += (words[w] + '_')
                merged_phrase = merged_phrase[:len(merged_phrase)-1]
                if len(merged_phrase) == 0:
                    continue
                if merged_phrase not in tf_idf[cnt_line]:
                    continue
                # if tf_idf[cnt_line][merged_phrase] < 0.5:
                #     continue
                line_phrase[merged_phrase] = tf_idf[cnt_line][merged_phrase]

        cnt_line += 1
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

        if len(docs_phrases[count_line]) > 2:
            sent_list.append(docs_phrases[count_line])
        count_line += 1
    docs_list.append(sent_list)

    # print(docs_list[41])
    return docs_list


if __name__ == "__main__":
    output_dir = '../../darkweb_data/4_30/'
    dict_phrases_file = output_dir + 'dict_sent_phrases_sg_3.txt'
    dict_phrases = open(dict_phrases_file, 'r')

    stopwords_file = open('../../darkweb_data/3_25/Stop_Words.txt', 'r')

    dictionary = getDictPhrases(dict_phrases)
    stopwords = getStopWords(stopwords_file)

    """ Unlabeled sampeles part """
    for idx in range(0, 11):
        inputData = open(output_dir + 'unlabeled_part/partitionOut_sg_3_unlabeled_' + str(idx)+ '.txt', 'r')

        # Load the trained word2vec model and the sentences
        sentences_data = open(output_dir + 'unlabeled_corpus/forum_40_input_phrases_indexed_unlabel_'
                                           + str(idx) + '.txt', 'r')

        tf_idf = pickle.load(open(output_dir + 'unlabeled_tfidf/tfidf_unlabeled_'
                                               + str(idx) + '.pickle', 'rb'))

        docs = createSentences(sentences_data, inputData, dictionary, stopwords, tf_idf)

        print(len(docs))
        pickle.dump(docs, open(output_dir + 'unlabeled_final/docs_' + str(idx) + '.pickle', 'wb'))

    """" Labeled samples part """
    # inputData = open(output_dir + 'partitionOut_sg_3.txt', 'r')
    #
    # # Load the trained word2vec model and the sentences
    # sentences_data = open(output_dir + 'forum_40_input_phrases_indexed.txt', 'r')
    #
    # tf_idf = pickle.load(open(output_dir + 'tfidf_f40_labels.pickle', 'rb'))
    #
    # docs = createSentences(sentences_data, inputData, dictionary, stopwords, tf_idf)
    #
    # print(len(docs))
    # pickle.dump(docs, open(output_dir + 'docs_label.pickle', 'wb'))
