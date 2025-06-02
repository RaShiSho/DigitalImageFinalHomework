import cv2
import numpy as np
import matplotlib.pyplot as plt

#定义图像数据的路径
img_path='histogram/data/boy.png'
img = cv2.imread(img_path, 0)
h, w = img.shape[:2]
out = np.zeros(img.shape, np.uint8)
#通过遍历对不同像素范围内进行分段线性变化
for i in range(h):
    for j in range(w):
        pix = img[i][j]
        if pix < 70:
            out[i][j] = 0.8 * pix
        elif pix < 180:
            out[i][j] = 2.6 * pix - 210
        else:
            out[i][j] = 0.138 * pix + 195
out = np.around(out)
out = out.astype(np.uint8)