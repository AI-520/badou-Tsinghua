# 最近邻插值
import cv2
import numpy as np
def function(img):
    height, width, channels = img.shape
    chang = input("长：")
    kuan = input("宽：")
    chang = int(chang)
    kuan = int(kuan)
    print(type(chang), type(kuan))
    qq = np.zeros((chang, kuan, channels), np.uint8)
    sh = chang/height
    sw = kuan/width
    for j in range(kuan):
        for i in range(chang):
            x = int(i/sh)
            y = int(j/sw)
            qq[i, j] = img[x, y]
    return qq
img = cv2.imread("C:/Users/AI/Pictures/Saved Pictures/20201229_200001.jpg")
ww = function(img)
print(ww.shape)
cv2.imshow("Linda", ww)
cv2.imshow("Mary", img)
cv2.waitKey(0)
