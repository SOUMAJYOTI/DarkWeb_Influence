import pickle
from textblob import TextBlob as tb
from tf_idf import tfidf


def getStopWords(data):
    for line in data:
        words = line.split(' ')
    # print(len(words))
    return words


def getInputPartitionedPhrases(data, stopwords):
    index_doc = 0
    doc = []
    for line in data:

        # for train crisis events data, append the doc number at front
        # line_phrase = data[index_doc]
        line_phrase = []
        index_doc += 1
        line = line[:len(line)-1]
        doc_phr = line.split(',')
        for ph in range(len(doc_phr)):
            phrases = doc_phr[ph]
            p = phrases.split(':')
            # count = dict[p[0]]
            words = p[0].split(' ')

            # separate conditions for single words and sentences !!!
            #1. Single word filter
            if len(words) < 2:
                # if count > 3 and (len(words[0]) > 3):
                if p[0] in stopwords:
                    continue
                line_phrase.append(p[0])
            #2. Phrase filter
            else:
                if True: #(count <= 500 and count > 2) or len(words) > 3:
                    merged_phrase = ''
                    for w in range(len(words)):
                        if words[w] in stopwords:
                            continue
                        merged_phrase += (words[w] + '_')
                    line_phrase.append(merged_phrase[:len(merged_phrase)-1])

        # if index_doc > 1000:
        #     break
        # print(line_phrase)
        doc.append(line_phrase)

    return doc


def calculateTfIdf(docs):
    bloblist = []
    for d in range(len(docs)):
        doc = docs[d]
        doc = doc[:len(doc)]
        doc_str = ''
        for w in range(len(doc)):
            doc_str += doc[w] + ' '
        doc_str = doc_str[:len(doc_str)-1]
        bloblist.append(tb(doc_str))

    ScoresDoc = [{} for _ in range(len(docs))]
    scores_dict = {}
    for i, blob in enumerate(bloblist):
        # print("Top words in document {}".format(i + 1))
        scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words:
            ScoresDoc[i][word] = score
            # print("Word: {}, TF-IDF: {}".format(word, round(score, 5)))

    return ScoresDoc


def getSentenceSplits(docs, stop):
    docs_split = []
    for line in docs:
        # print(line)
        sent = line[:len(line)-3]
        words = sent.split(' ')

        words_filter = []
        for w in range(len(words)):
            if words[w] in stop:
                continue
            words_filter.append(words[w])

        docs_split.append(words_filter)

    return docs_split

if __name__ == "__main__":

    print('Load files....')
    # output_dir = '../../darkweb_data/4_23/'

    # Stopwords
    stopwords_file = open('../../darkweb_data/Stop_Words.txt', 'r')
    stopwords = getStopWords(stopwords_file)

    """ Unlabeled samples part"""
    # for idx in range(0, 11):
    #     inputData = open(output_dir + 'unlabeled_part/partitionOut_sg_3_unlabeled_' + str(idx)+ '.txt', 'r')
    #
    #     # print(inputData)
    #     print('Start partitioning...')
    #     docs = getInputPartitionedPhrases(inputData, stopwords)
    #     print(len(docs))
    #     tfidfScore = calculateTfIdf(docs)
    #     print(len(tfidfScore))
    #
    #     pickle.dump(tfidfScore, open('../../darkweb_data/4_23/unlabeled_tfidf/tfidf_unlabaled_'
    #                                  + str(idx) + '.pickle', 'wb'))

    """  Labeled examples part """
    # inputData = open('../../darkweb_data/4_23/partitionOut_sg_3.txt', 'r')
    #
    # print('Start partitioning...')
    # docs = getInputPartitionedPhrases(inputData, stopwords)
    # print(len(docs))
    # tfidfScore = calculateTfIdf(docs)
    # print(len(tfidfScore))
    #
    # pickle.dump(tfidfScore, open('../../darkweb_data/4_23/tfidf_f40_labels.pickle', 'wb'))

    # tfidfScore = pickle.load(open('../../darkweb_data/04/4_23/tfidf_f40_labels.pickle', 'rb'))

    inputData = open('../../darkweb_data/05/5_19/forum40_sentences.txt', 'r')
    sent_splits = getSentenceSplits(inputData, stopwords)
    tfidfScore = calculateTfIdf(sent_splits)