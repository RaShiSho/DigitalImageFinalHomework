# 导入库
import numpy as np
import cv2

X = cv2.imread('/data/workspace/myshixun/stepB/A/image1.jpg', 0)

v = int(input())

# 定义 C
########## Begin ##########
C = 1.0 * v / np.log(1 + v)
##########  End  ##########

# 对 X 进行对数变换
########## Begin ##########
result = [C * np.log(1 + k) for k in X]
##########  End  ##########

# 由于 np.log() 函数会降低精度，所以将结果转换成高精度
result = np.array(result, np.float64) 

