import numpy as np
import cv2

# 读取 RGB 图像
X = cv2.imread('/data/workspace/myshixun/step7/A/image1.jpg', 1)
Y = cv2.imread('/data/workspace/myshixun/step7/A/image2.jpg', 1)

# 任务1：X,Y 进行乘法运算
########## Begin ##########
result = cv2.multiply(X, Y)
##########  End  ##########

# 将结果写入路径
cv2.imwrite("/data/workspace/myshixun/step7/C/result7.jpg", result)
print(np.sum(result))

# 任务1：X,Y 进行减法运算
########## Begin ##########
result = cv2.subtract(X, Y)
##########  End  ##########

# 将结果写入路径
cv2.imwrite("/data/workspace/myshixun/step6/C/result6.jpg", result)
print(np.sum(result))

# 任务1：对 X,Y 进行加法运算
########## Begin ##########
result = cv2.add(X, Y)
##########  End  ##########

# 将结果写入路径
cv2.imwrite("/data/workspace/myshixun/step5/C/result5.jpg", result)
print(np.sum(result))