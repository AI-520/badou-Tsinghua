# 使用K_Means算法实现图像像素的聚类
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('C:/Users/AI/Pictures/Saved Pictures/20201229_200001.jpg')

date = img.reshape(-1, 3)# 二位像素转为一维
date = np.float32(date)

ceitreia = (cv.TERM_CRITERIA_EPS +
            cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)# 设置停止条件

flags = cv.KMEANS_RANDOM_CENTERS# 设置标签
# 将图像聚成几类
compactness, labels2, centers2 = cv.kmeans(date, 2, None, ceitreia, 10, flags)
compactness, labels10, centers10 = cv.kmeans(date, 10, None, ceitreia, 10, flags)
compactness, labels20, centers20 = cv.kmeans(date, 20, None, ceitreia, 10, flags)
compactness, labels30, centers30 = cv.kmeans(date, 30, None, ceitreia, 10, flags)
compactness, labels40, centers40 = cv.kmeans(date, 40, None, ceitreia, 10, flags)
# 转换为unit8二位类型
centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]
dst2 = res.reshape((img.shape))

centers10 = np.uint8(centers10)
res = centers10[labels10.flatten()]
dst10 = res.reshape((img.shape))

centers20 = np.uint8(centers20)
res = centers20[labels20.flatten()]
dst20 = res.reshape((img.shape))

centers30 = np.uint8(centers30)
res = centers30[labels30.flatten()]
dst30 = res.reshape((img.shape))

centers40 = np.uint8(centers40)
res = centers40[labels40.flatten()]
dst40 = res.reshape((img.shape))
# 图像以rgb格式显示
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
dst2 = cv.cvtColor(dst2, cv.COLOR_BGR2RGB)
dst10 = cv.cvtColor(dst10, cv.COLOR_BGR2RGB)
dst20 = cv.cvtColor(dst20, cv.COLOR_BGR2RGB)
dst30 = cv.cvtColor(dst30, cv.COLOR_BGR2RGB)
dst40 = cv.cvtColor(dst40, cv.COLOR_BGR2RGB)
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 显示图像的名称
titles = [u'原始图像', u'聚类图像k=2', u'聚类图像k=10',
          u'聚类图像k=20', u'聚类图象k=30', u'聚类图象k=40']

images = [img, dst2, dst10, dst20, dst30, dst40]
for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray'),
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

