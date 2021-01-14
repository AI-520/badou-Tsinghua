# pca算法求出样本矩阵x的k阶降维矩阵z
import numpy as np
class CPCA(object):
    def __init__(self, x, k):# x为样本矩阵，k为降维矩阵的阶数
        self.x = x# 样本矩阵x
        self.k = k# k阶降维矩阵的k值
        self.centrx = []# 矩阵x的中心化
        self.c = []# 样本集的协方差矩阵c
        self.u = []# 样本矩阵x的将为转换矩阵
        self.z = []# 样本矩阵x的降维矩阵z
        self.centrx = self._centralized()
        self.c = self._cov()
        self.u = self._u()
        self.z = self._z()# z=xu求得

    def _centralized(self):# 矩阵x的中心化
        print('样本矩阵x:\n', self.x)
        centrx = []
        mean = np.array([np.mean(attr)for attr in self.x.T])# 样本集的特征均值
        print('样本集的特征均值:\n', mean)
        centrx = self.x - mean# 样本集的中心化
        print('样本矩阵x的中心化centrx:\n', centrx)
        return centrx

    def _cov(self):# 求样本矩阵x的协方差矩阵c
        ns = np.shape(self.centrx)[0]# 样本集的样例总数
        c = np.dot(self.centrx.T, self.centrx)/(ns-1)# 样本矩阵的协方差矩阵
        print('样本矩阵x的协方差矩阵c:\n', c)
        return c

    def _u(self):
        a, b = np.linalg.eig(self.c)# 将特征值赋值给a,特征向量赋值给b
        print('样本集协方差矩阵c的特征值\n:', a)
        print('样本集协方差矩阵c的特征向量\n:', b)
        ind = np.argsort(-1 * a)# 给出特征值降序的topk的索引序列
        ut = [b[:, ind[i]]for i in range(self.k)]
        u = np.transpose(ut)# 构建k阶降维的降维转换矩阵u
        print('%d阶降维转换矩阵u:\n' % self.k, u)
        return u

    def _z(self):
        z = np.dot(self.x, self.u)# 按照z=xu求出降维矩阵z
        print('x shape:', np.shape(self.x))
        print('u shape:', np.shape(self.u))
        print('self.z:', np.shape(z))
        print('样本矩阵x的降维矩阵z:\n', z)
        return z

if __name__ == '__main__':
    x = np.array([[49, 94, 20],
                  [67, 62, 65],
                  [23, 45, 83],
                  [32, 65, 32],
                  [90, 73, 84],
                  [58, 21, 83],
                  [73, 34, 54],
                  [83, 59, 26],
                  [29, 74, 32],
                  [72, 19, 20]])
    k = np.shape(x)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征)：\n', x)
    pca = CPCA(x, k)
