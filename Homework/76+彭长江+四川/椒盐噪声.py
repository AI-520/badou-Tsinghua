import numpy as np
import cv2 as cv
import random
from numpy import shape

def fun1(src, percetage):
    noiseimg = src
    noisenum = int(percetage * src.shape[0] * src.shape[1])
    '''
    每次取一个随机点
    图片像素中，rangx生成的行，rangy生成的列
    random.randint表示生成的随机整数
    '''
    for i in range(noisenum):
        # 椒盐噪声图片边缘不做处理，所以-1
        randx = random.randint(0, src.shape[0] - 1)
        randy = random.randint(0, src.shape[1] - 1)
        # random.random生成随机浮点数，随意取一个像素点，白点255，黑点0
        if random.random() <= 0.5:
            noiseimg[randx, randy] = 0
        else:
            noiseimg[randx, randy] = 255
    return noiseimg

img = cv.imread('C:/Users/AI/Pictures/Saved Pictures/20201229_200001.jpg', 0)
img1 = fun1(img, 0.2)

img = cv.imread('C:/Users/AI/Pictures/Saved Pictures/20201229_200001.jpg')
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('原图', img2)
cv.imshow('处理图', img1)
cv.waitKey(0)
