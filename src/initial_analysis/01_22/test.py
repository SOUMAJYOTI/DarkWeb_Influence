import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import spatial


# a = np.random.random((16, 16))
# plt.imshow(a, cmap='hot', interpolation='nearest')
# plt.show()

file_40 = pickle.load(open('02_01_2014_topicsDist.pickle', 'rb'))
file_84 = pickle.load(open('03_01_2015_topicsDist.pickle', 'rb'))

r = [[0 for _ in range(50)] for _ in range(50)]
for idx in range(len(file_40)):
    for idx_1 in range(len(file_84)):
        list_1 = sorted(file_40[idx], reverse=True)[:50]
        # print(file_40[idx])
        list_2 = sorted(file_84[idx_1], reverse=True)[:50]

        result = 1 - spatial.distance.cosine(list_1, list_2)
        r[idx][idx_1] = result

a = np.array(r)
# plt.imshow(a, cmap='hot', interpolation='nearest')
im = plt.matshow(a, cmap=plt.cm.hot, aspect='auto') # pl is pylab imported a pl
# plt.colorbar(im)
plt.show()
plt.show()