from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import cv2 as cv

x = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]
z = linkage(x, 'ward')
f = fcluster(z, 4, 'distance')
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(z)
print(z)
plt.show()