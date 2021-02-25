import cv2 as cv
import numpy as np

# 均值哈希算法
def ahash(img):
    # 将图片缩放成8*8
    img = cv.resize(img, (8, 8), interpolation=cv.INTER_CUBIC)
    # 将图片转换为灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # x为像素的最初值，
    x = 0
    # hash_str的最初值为‘’
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            x = x + gray[i, j]
    # 求平均速度
    avy = x / 64
    # 灰度值大于平均值就为1，小于就为0
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avy:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

# 差值感知算法
def dhash(img):
    # 将图片缩放成8*9
    img = cv.resize(img, (9, 8), interpolation=cv.INTER_CUBIC)
    # 将图片转换成灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素就为1， 反之就为0
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

# 哈希值对比
def cmphash(hash1, hash2):
    n = 0
    # 哈希长度不一样就返回-1，表示参数出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数加1，n为最终的相似度
        if hash1[i] != hash2[i]:
            n = n+1
    return n

img1 = cv.imread('C:/Users/AI/Pictures/Saved Pictures/A/20210222_080517.jpg')
img2 = cv.imread('C:/Users/AI/Pictures/Saved Pictures/A/20210222_080653.jpg')
hash1 = ahash(img1)
hash2 = ahash(img2)
print(hash1)
print(hash2)
n = cmphash(hash1, hash2)
print('均值哈希算法的相似度：', n)
hash1 = dhash(img1)
hash2 = dhash(img2)
print(hash1)
print(hash2)
n = cmphash(hash1, hash2)
print('插值算法的相似度：', n)
