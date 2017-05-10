import matplotlib.pyplot as plt
import numpy as np

def plot_bars(y, bar_width=1., x_label='', y_label='', col='#808080', rot=0, xTitles=[]):
    # plt.rc('text', usetex=True)
    # plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    # plt.rcParams['text.latex.preamble'] = [r'\boldmath']

    hfont = {'fontname': 'Arial'}
    plt.figure(figsize=(12, 8))
    x = np.arange(len(y))
    plt.bar(x, y, bar_width, color=col)

    if len(xTitles) > 0:
        major_ticks = np.arange(0, len(xTitles), 1)
        labels = []
        for i in major_ticks:
            labels.append(str(xTitles[i]))

        if rot != 0:
            plt.xticks(major_ticks, labels, rotation=rot, size=20)
        else:
            plt.xticks(major_ticks, labels, size=20)
    else:
        plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlabel(x_label, size=25, **hfont)
    plt.ylabel(y_label, size=25, **hfont)
    # plt.title('Month-wise post counts', size=20)

    plt.subplots_adjust(left=0.13, bottom=0.30, top=0.95)
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    y = [763.33, 404.54, 860.94, 1136.13, 1445.65, 926.60, 825.00]
    titles = ['Phone', 'Social Media', 'Email', 'Website',
              'Database', 'Grade Change', 'Other']
    ylabel='Mean Price(USD)'
    xlabel='Service Request Category'
    plot_bars(y, bar_width=0.5, xTitles=titles, rot=45, x_label=xlabel, y_label=ylabel)
