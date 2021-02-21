import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()
# 特征空间的4个维度
x = iris.data[:, :4]
print(x.shape)
dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(x)
label_pred = dbscan.labels_

x0 = x[label_pred == 0]
x1 = x[label_pred == 1]
x2 = x[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c='red', marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c='green', marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c='blue', marker='+', label='label2')
plt.xlabel('1')
plt.ylabel('2')
plt.legend(loc=2)
plt.show()
