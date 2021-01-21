import numpy as np

def  WarpPerspectveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    
    nums = src.shape[0]
    a = np.zeros((2 * nums, 8))
    b = np.zeros((2 * nums, 1))
    for i in range(0, nums):
        a_i = src[i, :]
        b_i = dst[i, :]
        a[2 * i:] = [a_i[0], a_i[1], 1, 0, 0, 0,
                     -a_i[0] * b_i[0], -a_i[1] * b_i[0]]
        b[2 * i] = b_i[0]

        a[2 * i + 1, :] = [0, 0, 0, a_i[0], a_i[1], 1,
                        -a_i[0] * b_i[1], -a_i[1] * b_i[1]]
        b[2 * i + 1] = b_i[1]
    a = np.mat(a)
    warpMatrix = a.I * b
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)
    warpMatrix = warpMatrix.reshape(3, 3)
    return warpMatrix

if __name__=='__main__':
    print('WarpMatrix')
    src = [[150.0, 457.0], [395.0, 363.0], [633.0, 291.0], [766.0, 457.0]]
    src = np.array(src)
    dst = [[46.0, 124.0], [46.0, 654.0], [634.0, 363.0], [466.0, 436.0]]
    dst = np.array(dst)
    warpMatrix = WarpPerspectveMatrix(src, dst)
    print(warpMatrix)
