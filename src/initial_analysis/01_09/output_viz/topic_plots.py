# N.B. : the intervals stored in the pickle files are reverse in order.
# reverse the order in this program to get the original order of the intervals.

# this is the general utility plot program
__author__ = 'ssarka18'

import pandas as pd
from pylab import *
import os
import math
import statistics as st
import statistics
from math import  *
import datetime
import pickle
from pylab import *
import operator

months_topics = pickle.load(open('sim_months_topics.pickle', 'rb'))

for d in months_topics:
    data_to_plot = []
    sorted_t = sorted(months_topics[d].items(), key=operator.itemgetter(0))
    labels = []
    for t, l in sorted_t:
        data_to_plot.append(l)
        labels.append(t)

    fig = plt.figure(1, figsize=(10, 8))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(data_to_plot, patch_artist=True)

    for box in bp['boxes']:
        # change outline color
        box.set(color='#0000FF', linewidth=2)
        # change fill color
        box.set(facecolor='#FFFFFF')

        ## change color and linewidth of the whiskers
        # for whisker in bp['whiskers']:
        #     whisker.set(color='#7570b3', linewidth=2)

        ## change color and linewidth of the caps
        # for cap in bp['caps']:
        #     cap.set(color='#7570b3', linewidth=2)

        ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#FF0000', linewidth=4)

        ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    third_quartile = [item.get_ydata()[0] for item in bp['whiskers']]
    third_quartile = max(third_quartile)

    first_quartile = [item.get_ydata()[1] for item in bp['whiskers']]
    first_quartile = max(first_quartile)


    ax.set_title('Similarity score between users and out-neighbors', fontsize=15)
    #ax.set_title(r'\textbf{Shortest path - Newly appeared nodes by interval}', fontsize=55)
    ax.set_xlabel('Topics', fontsize=25)
    ax.set_ylabel('Similarity score', fontsize=25)


    # plt.ylim([-third_quartile - 0.5*math.pow(10, int(math.log10(third_quartile))),
    #           third_quartile + math.pow(10, int(math.log10(third_quartile)))])
    # plt.ylim([0, third_quartile + math.pow(10, int(math.log10(third_quartile)))])
    plt.ylim([0.9, 1.0])
    plt.tick_params('y', labelsize=20)
    plt.tick_params('x', labelsize=20)
    plt.grid(True)
    # ax.set_xticklabels(titles)
    plt.savefig('users_topics_sim/' + str(d) + ".png")
    plt.close()