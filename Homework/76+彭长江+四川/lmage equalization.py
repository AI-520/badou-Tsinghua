# 图像均衡化
import cv2
import numpy as np
from matplotlib import pyplot as plt
# 获取图像
img = cv2.imread("C:/Users/AI/Pictures/Saved Pictures/20201229_200001.jpg", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 调用函数equalizeHist
dst = cv2.equalizeHist(gray)
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()
cv2.imshow("Mray", np.hstack([gray, dst]))
cv2.waitKey(0)
