import pickle
import matplotlib.pyplot as plt
import numpy as np
# def getDocs():

if __name__ == "__main__":
    fId = 40
    output_dir = '../../../darkweb_data/2_2/nlp_process/forum/' + str(fId) + '/phrases_months/all/'
    tfidf_score = pickle.load(open(output_dir + 'tfIDf_score.pickle', 'rb'))

    Scores = []
    for doc in tfidf_score:
        for s in doc:
            Scores.append(doc[s])

    # print(np.count_nonzero(np.array(Scores) > 0.15))
    # plt.figure(figsize=(12, 8))
    # hfont = {'fontname': 'Arial'}
    # n, bins, patches = plt.hist(Scores, 50, facecolor='b')
    # # plt.yscale('log', nonposy='clip', basey=2)
    # plt.xlabel('Tf-Idf score', size=40, **hfont)
    # plt.ylabel('Frequency', size=40, **hfont)
    # # plt.title('Histogram of')
    # # plt.xlim([0, 700])
    # # plt.ylim([0, 2 ** (12)])
    # plt.grid(True)
    # plt.xticks(size=25)
    # plt.yticks(size=25)
    # # file_save = dir_save + '/' + 'count_motif_' + str(m) + '.png'
    # # plt.savefig(file_save)
    # plt.subplots_adjust(left=0.16, bottom=0.16)
    # plt.show()