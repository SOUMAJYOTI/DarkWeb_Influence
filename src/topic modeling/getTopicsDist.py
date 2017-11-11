import fileinput
import random
import scipy.special
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import beta
import pprint, pickle
import datetime
import time


def VisualizeTopics(phi, words, num_topics, viz_threshold=9e-3):
    phi_viz = np.transpose(phi)
    words_to_display = ~np.all(phi_viz <= viz_threshold, axis=1)
    words_viz = [words[i] for i in range(len(words_to_display)) if words_to_display[i]]
    phi_viz = phi_viz[words_to_display]

    fig, ax = plt.subplots()
    heatmap = plt.pcolor(phi_viz, cmap=plt.cm.Blues, alpha=0.8)
    plt.colorbar()

    # fig.set_size_inches(8, 11)
    ax.grid(False)
    ax.set_frame_on(False)

    ax.set_xticks(np.arange(phi_viz.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(phi_viz.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    # plt.xticks(rotation=45)

    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    column_labels = words_viz  # ['Word ' + str(i) for i in range(1,1000)]
    row_labels = ['Topic ' + str(i) for i in range(1, num_topics + 1)]
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)

    plt.show()


def VisualizeEvolution(psi):
    xs = np.linspace(0, 1, num=1000)
    fig, ax = plt.subplots()

    for i in range(len(psi)):
        ys = [math.pow(1 - x, psi[i][0] - 1) * math.pow(x, psi[i][1] - 1) / scipy.special.beta(psi[i][0], psi[i][1]) for
              x in xs]
        ax.plot(xs, ys, label='Topic ' + str(i + 1))

    ax.legend(loc='best', frameon=False)
    plt.show()


def main():
    fId = 40
    month = 'all'
    date = '02_01_2014'
    resultspath = '../../../darkweb_data/2_2/results/forum/' + str(fId) + '/v1'
    tot_pickle_path = resultspath + '/' + month + '_tot.pickle'
    # dictionary = pickle.load(open(resultspath + '/dict.pickle', 'rb'))
    tstamps = []
    timestamps_file = '../../../darkweb_data/2_2/nlp_process/forum/' + str(
        fId) + '/phrases_months/v1' + '/timestamps.txt'
    for t in open(timestamps_file, 'r'):
        tstamps.append(t)

    start = '2014-02-01 00:00:00'
    end = '2014-03-01 00:00:00'
    time_struct = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    start_date = time.mktime(time_struct.timetuple())

    time_struct = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
    end_date = time.mktime(time_struct.timetuple())

    tot_pickle = open(tot_pickle_path, 'rb')
    par = pickle.load(tot_pickle)
    # for p in par:
        # print(p)
        # print(par[p])
    # VisualizeTopics(par['n'], par['word_token'], par['T'])
    # VisualizeEvolution(par['psi'])

    theta = par['m']
    phi = par['n']

    # print(len(tstamps))
    temp_t_w = [[0. for _ in range(len(phi[0]))] for _ in range(50)]
    sumTopics = [0. for _ in range(len(phi))]

    # print(len(phi[0]))
    # temp_t_w = np.zeros((50, len(phi)))
    # print(len(probTopics))
    cnt = 0
    for d in range(len(theta)):
        print(d)
        tstamps[d] = float(tstamps[d][:len(tstamps[d])-1])
        if tstamps[d] >= start_date and tstamps[d] < end_date:
            # print('hello')
            cnt += 1
            for topic in range(len(theta[d])):
                for word in range(len(phi[topic])):
                    # sumTopics[idx] += theta[d][idx]
                    # print(word)
                    temp_t_w[topic][word] += theta[d][topic] * phi[topic][word]
                    # print(theta[d][topic] * phi[topic][word])


    for t in range(len(temp_t_w)):
        for w in range(len(temp_t_w[t])):
            temp_t_w[t][w] /= len(theta)

    # normV = 0.
    # for t in range(len(sumTopics)):
    #     normV += sumTopics[t]
    #
    # for t in range(len(sumTopics)):
    #     sumTopics[t] /= normV
    #
    pickle.dump(temp_t_w, open(resultspath + '/' + date + '_topicsDist.pickle', 'wb'))


if __name__ == "__main__":
    main()
