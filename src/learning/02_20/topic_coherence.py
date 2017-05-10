import numpy as np
import pickle



def loadTopicWords(doc):
    phrases = []
    topic_phrase_dict = {}
    for line in doc:
        line = line[:len(line) - 1]
        if line[:5] == "Topic":
            t = line[:len(line) - 1]
            phrases = []
            continue
        elif line == '\n':
            continue
        elif line == '':
            topic_phrase_dict[t] = phrases
        else:
            line = line.replace(" ", "")
            w = line.split(':')
            if w[0] == 'K':
                break
            if w[0] == '':
                continue
            phrases.append(w[0])

    return topic_phrase_dict


def calculateCoOcurrence(docs, w1, w2):
    # print(len(docs))
    countOccur = 0
    countCoOccur = 0
    for d in docs:
        if (w1 in d) :
            countOccur += 1
            if w2 in d:
                countCoOccur += 1

    return (countOccur, countCoOccur)

def calculateProbOccur(docs, w):
    countOccur = 0
    for d in docs:
        if w in d:
            countOccur += 1

    return countOccur/len(docs)

def calculateTopicCoherence(topic_1, topic_2, docs):
    TopicCoherence = 0
    for word_1 in topic_1:
        for word_2 in topic_2:
            # coOccurProb = calculateCoOcurrence(docs, word_1, word_2)
            prob_w1,  coOccurProb = calculateCoOcurrence(docs, word_1, word_2)
            # prob_w2 = calculateProbOccur(docs, word_2)

            # print(coOccurProb, prob_w1)
            if prob_w1 == 0.:
                continue
                # np.log
            val = np.log((coOccurProb)/(prob_w1))
            # print(val)
            if np.isinf(val):
                continue
            TopicCoherence += val
            # print(coOccurProb, prob_w1, prob_w2)
            # print(word_1, word_2, TopicCoherence)

    # exit()
    return TopicCoherence

if __name__ == "__main__":
    forumsId = [40]
    forumsId_2 = [84]
    # date = '02_01_2014'
    dates_2= ['directive']
    dates_1 = ['sarcastic']

    topicCoherence = {}

    for f in range(len(forumsId)):
        output_dir = './phrases_sarcastic'
        forumDoc_File = open('./phrases_sensemaking/posts.txt', 'r')
        forumDoc = []
        for line in forumDoc_File:
            line = line[:len(line)-1]
            words = line.split(' ')
            forumDoc.append(words)

        output_dir_1 = './phrases_directive'
        forumDoc_File = open(
            './posts.txt', 'r')
        for line in forumDoc_File:
            line = line[:len(line) - 1]
            words = line.split(' ')
            forumDoc.append(words)

        for d1 in range(len(dates_1)):
            topicFile_1 = open(output_dir + '/output_phrases.txt', 'r')
            topicWords_1 = loadTopicWords(topicFile_1)

            if dates_1[d1] not in topicCoherence:
                topicCoherence[dates_1[d1]] = {}
            # print(topicWords['Topic 1'])
            for d2 in range(len(dates_2)):
                print(dates_1[d1], dates_2[d2])
                topicCoherence[dates_1[d1]][dates_2[d2]] = {}
                topicFile_2 = open(output_dir_1 + '/output_phrases.txt', 'r')
                topicWords_2 = loadTopicWords(topicFile_2)

                for topic_1 in topicWords_1:
                    topicCoherence[dates_1[d1]][dates_2[d2]][topic_1] = {}
                    tw1 = topicWords_1[topic_1]
                    for topic_2 in topicWords_2:
                        tw2 = topicWords_2[topic_2]

                        coherence = calculateTopicCoherence(tw1, tw2, forumDoc)
                        print(topic_1, topic_2, coherence)
                        topicCoherence[dates_1[d1]][dates_2[d2]][topic_1][topic_2] = coherence


        pickle.dump(topicCoherence, open('sarcastic_directive_tc.pickle', 'wb'))

