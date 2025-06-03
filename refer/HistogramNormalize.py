import cv2
import numpy as np

#读取原图片
img_ori=cv2.imread(r'/data/workspace/myshixun/原图/image.png',flags=0)
#读取目标图片
object=cv2.imread(r'/data/bigfiles/049692cf-7281-4eba-b0fc-9118621ba670',flags=0)

img=img_ori.copy()
scr=object.copy()

mHist1=[]
mNum1=[]
inhist1=[]
mHist2=[]
mNum2=[]
inhist2=[]


# 对原图像进行均衡化
for i in range(256):
    mHist1.append(0)

# 获取原图像像素点的宽度和高度
row, col = img.shape
for i in range(row):
    for j in range(col):
        mHist1[img[i, j]] = mHist1[img[i, j]] + 1

mNum1.append(mHist1[0] / img.size)
for i in range(0, 255):
    mNum1.append(mNum1[i] + mHist1[i + 1] / img.size)
for i in range(256):
    inhist1.append(round(255 * mNum1[i]))


# 对目标图像进行均衡化
for i in range(256):
    mHist2.append(0)
rows, cols = scr.shape 
for i in range(rows):
    for j in range(cols):
        mHist2[scr[i, j]] = mHist2[scr[i, j]] + 1 
mNum2.append(mHist2[0] / scr.size)
for i in range(0, 255):
    mNum2.append(mNum2[i] + mHist2[i + 1] / scr.size)
for i in range(256):
    inhist2.append(round(255 * mNum2[i]))
    

# 获取目标图像像素点的宽度和高度


# 进行规定化
# 用于放入规定化后的图片像素
g = []
for i in range(256):
    a = inhist1[i]
    flag = True
    for j in range(256):
        if inhist2[j] == a:
            g.append(j)
            flag = False
            break
    if flag == True:
        minp = 255
        for j in range(256):
            b = abs(inhist2[j] - a)
            if b < minp:
                minp = b
                jmin = j
        g.append(jmin)
for i in range(row):
    for j in range(col):
        img[i, j] = g[img[i, j]] 


cv2.imwrite("/data/workspace/myshixun/学员文件/img.jpg",img)
# 打印结果
print(np.sum(img)) 

