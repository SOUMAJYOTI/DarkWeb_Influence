import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

#here's our data to plot, all normal Python lists
x = range(20)
y = range(20)

import pickle
tc = pickle.load(open('f40_84_tc.pickle', 'rb'))

tc_topics_map = [[0 for _ in range(20)] for _ in range(20)]
for d1 in tc:
    for d2 in tc[d1]:
        tc_topics = tc[d1][d2]
        # print(tc_topics)

cnt_t1 = -1
cnt_t2 = -1
for t1 in tc_topics:
    cnt_t1 += 1
    cnt_t2 = -1
    for t2 in tc_topics:
        cnt_t2 += 1
        # print(cnt_t1, cnt_t2)
        tc_topics_map[cnt_t1][cnt_t2] = tc_topics[t1][t2]


intensity = tc_topics_map
print(intensity)
# intensity = [
#     [5, 10, 15, 20, 25],
#     [30, 35, 40, 45, 50],
#     [55, 60, 65, 70, 75],
#     [80, 85, 90, 95, 100],
#     [105, 110, 115, 120, 125]
# ]

intRot = np.flipud(intensity)
# intRot = np.rot90(intRot, 2)
# print(intRot)
#setup the 2D grid with Numpy
x, y = np.meshgrid(x, y)

#convert intensity (list of lists) to a numpy array for plotting
intensity = np.array(intRot)

#now just plug the data into pcolormesh, it's that easy!
plt.pcolormesh(x, y, cmap=cm.gray, vmin=-200, vmax=-1500)
cbar = plt.colorbar() #need a colorbar to show the intensity scale

plt.tick_params('x', labelsize=30)
plt.tick_params('y', labelsize=30)
plt.ylabel('02/01/2014 (1 month) - Forum 40', fontsize=30)
plt.xlabel('08/01/2015 (1 month) - Forum 84', fontsize=30)
cbar.ax.tick_params(labelsize=30)

# plt.xlabel('', fontsize=30)

plt.show() #boom