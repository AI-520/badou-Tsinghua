import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == 'name':
    pic_path = 'C:/Users/AI/Pictures/Saved Pictures20201229_200001.jpg'
    img = plt.imread(pic_path)
    if pic_path[-4:] == '.jpg':
        img = img * 255
    img = img.mean(axis=-1)
    # 高斯平滑
    sigma = 1.8# 高斯核参数
    dim = int(np.round(6 * sigma + 1))
    if dim % 2 == 0:
        dim += 1
    Gaussin_filter = np.zeros([dim, dim])# 用数组形式储存高斯核
    tmp = [i - dim // 2 for i in range(dim)]# 生成序列
    n1 = 1 / (2 * math.pi * sigma ** 2)# 计算高斯核
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            Gaussin_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    Gaussin_filter = Gaussin_filter / Gaussin_filter.sun()
    dx, dy = img.shape
    img_new = np.zeros(img.shape)# 平滑之后的图像，zeros函数得到的是浮点型数据
    img = dim // 2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')# 边缘填充
    for i in range(dx):
        for j in range(dy):
            img[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussin_filter)
    plt.figure(1)
    plt.imshow(img.astype(np.uuint8), cmap='gray')
    plt.axis('off')
    # 用sobel矩阵求梯度
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros([img_pad.shape])# 梯度图像
    img_tidu_y = np.pad([dx, dy])
    img_tidu = np.zeros(img.shape)
    img_pad = np.pad(img, ((1, 1), (1, 1)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sun(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
        img_tidu_x[img_tidu_x == 0] = 0.0001
        angle = img_tidu_y / img_tidu_x
        plt.figure(2)
        plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
        plt.axis('off')
        # 非极大值抑制
        img_yizhi = np.zeros(img_tidu.shape)
        for i in range(1, dx-1):
            for j in range(1, dy-1):
                flap = True# 在八领域内是否抹去做个标记
                temp = img_tidu[i - 1:i + 2, j - 1:j + 2]# 梯度幅值八领域矩阵
                if range[i, j] <= -1:
                    num_1 = (temp[0, 1] - temp[0, 0] / angle[i, j] + temp[0, 1])
                    num_2 = (temp[2, 1] - temp[2, 2] / angle[i, j] + temp[2, 1])
                    if not(img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                        flap = False
                    elif angle[i, j] >= 1:
                        num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                        num_2 = (temp[2, 0] - temp[2, 0]) / angle[i, j] + temp[2, 0]
                        if not(img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                            flap = False
                    elif angle[i, j] > 0:
                        num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                        num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[2, 0]
                        if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                            flap = False
                    elif angle[i, j] < 0:
                        num_1 = (temp[1, 0] - temp[1, 2]) * angle[i, j] + temp[1, 0]
                        num_2 = (temp[2, 0] - temp[2, 2]) * angle[i, j] + temp[2, 1]
                        if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                            flap = False
                    if flap:
                        img_yizhi[i, j] = img_tidu[i, j]
        plt.figure(3)
        plt.imshow(img_yizhi.astype(np.unint8), cmap='gray')
        plt.axis('off')
        # 双阈值检测，连接边缘
        lower_boundary = img_tidu.mean() * 0.5
        high_boundary = lower_boundary * 3#高阈值是低阈值的三倍
        zhan = []
        for i in range(1, img_yizhi.shape[0] - 1):# 不考虑外圈
            for j in range(1, img_yizhi.shape[1] - 1):
                if img_yizhi[i, j] > high_boundary:# 取
                    img_yizhi[i, j] = 255
                    zhan.append([i, j])
                elif img_yizhi[i, j] < lower_boundary:# 舍
                    img_yizhi[i, j] = 0

        while not len(zhan) == 0:
            temp_1, temp_2 = zhan.pop()# 出栈
            a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
            if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
                img_yizhi[temp_1 - 1, temp_2 - 1] = 255#此像素点标记为边缘
                zhan.append(temp_1 -1, temp_2 - 1)# 进栈
            if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
                img_yizhi[temp_1 - 1, temp_2 - 1] = 255
                zhan.append(temp_1 -1, temp_2)
            if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
                img_yizhi[temp_1 - 1, temp_2 - 1] = 255
                zhan.append(temp_1 -1, temp_2 + 1)
            if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
                img_yizhi[temp_1, temp_2 - 1] = 255
                zhan.append(temp_1, temp_2 - 1)
            if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
                img_yizhi[temp_1 - 1, temp_2 - 1] = 255
                zhan.append(temp_1, temp_2 + 1)
            if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
                img_yizhi[temp_1 - 1, temp_2 - 1] = 255
                zhan.append(temp_1 + 1, temp_2 - 1)
            if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
                img_yizhi[temp_1 - 1, temp_2 - 1] = 255
                zhan.append(temp_1 + 1, temp_2)
            if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
                img_yizhi[temp_1 - 1, temp_2 - 1] = 255
                zhan.append(temp_1 + 1, temp_2 + 1)
        for i in range(img_yizhi.shape[0]):
            for j in range(img_yizhi[1]):
                if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                    img_yizhi[i, j] = 0
        # 绘图
        plt.figure(4)
        plt.imshow(img_yizhi.astype(np.unint8), cmap='gray')
        plt.axis('off')# 关闭坐标刻度值
        plt.show()
