import datetime as dt
import pandas as pd
import requests
import operator
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import collections
import datetime
import time
from collections import Counter
import calendar
import calendar
from bs4 import BeautifulSoup
import re
import os
import pickle


def getStopWords(data):
    for line in data:
        words = line.split(' ')
    # print(len(words))
    return words


def getSentences(content):
    sent_list = content.split('.')
    # print(sent_list)

    # Now filter some sentences which are actually not period aborted sentences
    new_sent_list = []
    for s in sent_list:
        if len(s) < 10:
            continue
        new_sent_list.append(s.lstrip())

    if len(new_sent_list) == 0:
        return []

    process_sent_list = []
    for s in new_sent_list:
        # print(s)
        cont = s
        try:
            cont = BeautifulSoup(cont).get_text()
            # print(cont)
        except:
            return ''

        cont = re.sub(r'http\S+', '', cont)
        # cont = cont[:len(cont)-1]
        cont = re.sub("[^a-zA-Z']",  # The pattern to search for
                      " ",  # The pattern to replace it with
                      cont)  # The text to search

        cont = " ".join(cont.split())
        words = cont.split(' ')
        temp = ''
        for w in range(len(words)):
            if re.search(r'((\w)\2{2,})', words[w]):
                continue

            if "aa" in words[w] or "xx" in words[w] or "yy" in words[w] or \
                            "zz" in words[w] or "yx" in words[w] or "zx" in words[w] or "xz" in \
                    words[w]:
                continue
            if len(words[w]) < 15 and len(words[w]) > 2 and words[w] != 'quote':
                temp += (words[w] + ' ')
            # if words[w] == 'quote' and (w+1 < len(range(len(words)))):
            #     words[w+1] = ''
        if temp == '':
            # print('hello', cont_temp)
            continue
        # temp = cont
        # if temp[0] == ' ':
        #     temp = temp[1:]
        temp = temp[:len(temp)-1]
        process_sent_list.append(temp)
    # print(process_sent_list)
    return process_sent_list


def getCorpusPosts(fPosts, timecheck=True):
    mdocs = []
    mdocs_indexed = []
    idx_l = 0

    docPosts = {}

    for line in fPosts:
        line_decode = line.decode('ISO-8859-1')
        temp = getSentences(line_decode)
        if temp == []:
            continue
        docPosts[idx_l] = temp
        # print(line.decode('ISO-8859-1'))
        idx_l += 1

        # if len(docPosts) > 3:
        #     break

    cnt_doc = 0
    # Keep two DS - one for indexing the sentence with the document number
    # and the other for phrase partitioning in Apache Spark Java
    for d in docPosts:
        sent_list = docPosts[d]
        for cont in sent_list:
            temp = ''
            temp_idx = str(d) + ' '
            words = cont.split(' ')
            if len(words) < 3:
                continue
            for w in range(len(words)):
                # countW_Docs = wordsinPostsCount(words[w], docPosts)
                # if countW_Docs >= 10:
                    temp += (words[w] + ' ')
            if temp == '' or temp == ' ':
                continue
            temp = temp[:len(temp)-1] + '.'
            temp_idx += temp
            cnt_doc += 1
            # print(cnt_doc)
            # words = temp.split(' ')
            # if len(words) <= 3:
            #     continue
            mdocs.append(temp)
            mdocs_indexed.append(temp_idx)


        # if cnt_doc > 5:
        #     break

    return (mdocs, mdocs_indexed)


def getSentenceSplits(sent, stop):
    # Sentences with length less than 10 are not considered
    # Duplicate sentences are removed

    docs = []
    seen = []
    for idx_s in range(len(sent)):
        s = sent[idx_s]
        s = s[:len(s)-1]
        if s in seen:
            continue
        seen.append(s)
        words = s.split(' ')
        sent_filtered = []
        for idx_w in range(len(words)):
            if words[idx_w] not in stop:
                sent_filtered.append(words[idx_w])

        if len(sent_filtered) < 5:
            continue
        # print(sent_filtered)
        docs.append(sent_filtered)
    return docs


def createInpOutMatrix(sent, w2v_feat, sen_length):
    contextOutMatrix = []
    contextInpMatrix = []
    for s in range(len(sent)):
        context_input = []
        context_output = []
        sentence = sent[s]
        # Create the input
        for w in range(len(sentence)):
            if sentence[w] in w2v_feat:
                context_input.append(w2v_feat[sentence[w]])
            else:
                context_input.append(np.random.uniform(-0.25, 0.25, (50)))

        # Sentences should be of length 15
        if len(context_input) < sen_length:
            len_cur = len(context_input)
            for idx_fill in range(sen_length - len_cur):
                context_input.append(np.random.uniform(-0.25, 0.25, (50)))
        else:
            context_input = context_input[:sen_length]

        contextInpMatrix.append(context_input)

        # Create the output
        # Add the first word context for that sentence
        cnt_words = 0
        cur_idx = 1
        while cur_idx < len(sentence):
            if sentence[cur_idx] in w2v_feat:
                context_output.append(w2v_feat[sentence[cur_idx]])
                cnt_words += 1
                break
            cur_idx += 1

        if cnt_words == 0:
            context_output.append(np.random.uniform(-0.25, 0.25, (50))) # Sample from uniform distribution

        # Words from 1 to last but one
        for w in range(1, len(sentence)-1):
            word_left = sentence[w-1]
            word_right = sentence[w+1]
            sum = np.array([0. for _ in range(50)])
            cnt_words = 0
            if word_left in w2v_feat:
                sum += w2v_feat[word_left]
                cnt_words += 1
            if word_right in w2v_feat:
                sum += w2v_feat[word_right]
                cnt_words += 1

            if cnt_words == 2:
                sum_mean = sum / 2.
            else:
                sum_mean = sum

            # if both words do not exist
            if cnt_words == 0:
                cur_idx = w+2
                while cur_idx < len(sentence):
                    if sentence[cur_idx] in w2v_feat:
                        sum += w2v_feat[sentence[cur_idx]]
                        cnt_words += 1
                        break
                    cur_idx += 1

                cur_idx = w - 2
                while cur_idx >= 0:
                    if sentence[cur_idx] in w2v_feat:
                        sum += w2v_feat[sentence[cur_idx]]
                        cnt_words += 1
                        break
                    cur_idx -= 1

                if cnt_words == 2:
                    sum_mean = sum / 2.
                else:
                    sum_mean = sum

            if cnt_words == 0:
                sum_mean = np.random.uniform(-0.25, 0.25, (50)) # Sample from uniform distribution

            context_output.append(sum_mean)

        # Add the last word context for that sentence
        cnt_words = 0
        cur_idx = len(sentence) - 2
        while cur_idx >= 0:
            if sentence[cur_idx] in w2v_feat:
                context_output.append(w2v_feat[sentence[cur_idx]])
                cnt_words += 1
                break
            cur_idx -= 1

        if cnt_words == 0:
            context_output.append(np.random.uniform(-0.25, 0.25, (50))) # Sample from uniform distribution

        # Sentences should be of length 15
        if len(context_output) < sen_length:
            len_cur = len(context_output)
            for idx_fill in range(sen_length-len_cur):
                context_output.append(np.random.uniform(-0.25, 0.25, (50)))
        else:
            context_output = context_output[:sen_length]

        # print(np.array(context_output).shape)
        contextOutMatrix.append(context_output)

    contextOutMatrix = np.array(contextOutMatrix)
    contextInpMatrix = np.array(contextInpMatrix)
    return contextInpMatrix, contextOutMatrix


if __name__ == "__main__":
    stopwords_file = open('../../darkweb_data/Stop_Words.txt', 'r')
    stopwords = getStopWords(stopwords_file)
    w2v_feat = pickle.load(open('../../darkweb_data/5_15/word2vec_train_model_d50_min2.pickle', 'rb'))

    # sentences_all = []
    # for idx in range(1, 11):
    #     print(idx)
    #     forumsData = open('../../darkweb_data/4_23/unlabeled_data/unlabeled_' + str(idx) + '.txt', 'rb')
    #
    #     sentences = []
    #
    #     corpus, corpus_indexed = getCorpusPosts(forumsData)
    #     sentences = getSentenceSplits(corpus, stopwords)
    #     sentences_all.extend(sentences)
    #
    # pickle.dump(sentences_all, open('../../darkweb_data/sentences_unlabel.pickle', 'wb'))

    """ Load the unlabeled sentences"""
    sentence_max_len = 30
    sentences_all = pickle.load(open('../../darkweb_data/sentences_unlabel.pickle', 'rb'))
    InpMatrix, OutMatrix = createInpOutMatrix(sentences_all, w2v_feat, sentence_max_len)

    print(InpMatrix.shape, OutMatrix.shape)

    pickle.dump((InpMatrix, OutMatrix), open('../../darkweb_data/5_15/unlabeled/Input_Output_Matrices.pickle', 'wb'))

