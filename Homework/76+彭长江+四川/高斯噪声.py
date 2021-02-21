import numpy as np
import cv2 as cv
from numpy import shape
import random

def GaussianNoise(src, means, sigma, percetage):
    noiseimg = src
    noisenum = int(percetage * src.shape[0] * src.shape[1])
    # 每次取一个随机点
    for i in range(noisenum):
        '''
        图片像素中，randx代表行，randy代表列
        random.randit表示生成随机整数
        在高斯噪声中图片不做边缘处理，所以-1
        '''
        randx = random.randint(0, src.shape[0] - 1)
        randy = random.randint(0, src.shape[1] - 1)
        # 在原有像素灰度值上加上随机数
        noiseimg[randx, randy] = noiseimg[randx, randy] + random.gauss(means, sigma)
        # 判断灰度值的大小，小于0则为0，大于255则为255
        if noiseimg[randx, randy] < 0:
            noiseimg[randx, randy] = 0
        elif noiseimg[randx, randy] < 255:
            noiseimg[randx, randy] = 255
    return noiseimg
img = cv.imread('C:/Users/AI/Pictures/Saved Pictures/20201229_200001.jpg', 0)
img1 = GaussianNoise(img, 2, 4, 0.8)
img = cv.imread('C:/Users/AI/Pictures/Saved Pictures/20201229_200001.jpg')
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('原图', img2)
cv.imshow('处理图', img1)
cv.waitKey(0)
