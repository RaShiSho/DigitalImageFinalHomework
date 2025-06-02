import cv2 
import numpy as np

img_path = '/data/workspace/myshixun/原图/image.png'
########Begin########
# 图像以灰度图输入
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 图像放缩
src = cv2.resize(img, (256, 256))

#图像的直方图
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# 灰度图均衡化
equ = cv2.equalizeHist(src)

########End########
cv2.imwrite("/data/workspace/myshixun/学员文件/equ.jpg",equ)
# 打印结果
print(np.sum(equ)) 