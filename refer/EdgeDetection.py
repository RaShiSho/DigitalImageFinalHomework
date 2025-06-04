# -*- coding: utf-8 -*-
import cv2
import numpy as np

def robs():
    filepath = '/data/workspace/myshixun/task2/'
    # 读取图像
    img = cv2.imread(filepath+'road.png')
    ########## Begin ##########
    # 1. 灰度化处理图像
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2. Roberts算子
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    # 3. 卷积操作
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
    # 4. 数据格式转换
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    ########## End ##########
    
    # 保存图像
    cv2.imwrite(filepath+"out/roberts.png",Roberts)

def sob():
    filepath = '/data/workspace/myshixun/task3/'
    
    # 读取图像
    img = cv2.imread(filepath+'road.png')
    ########## Begin ##########
    # 1. 灰度化处理图像
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2. 求Sobel 算子
    kernelx = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0)
    kernely = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1)
    # 3. 数据格式转换
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    # 4. 组合图像
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    ########## End ##########
    # 保存图像
    cv2.imwrite(filepath+"out/sobel.png",Sobel)

def lap():
    filepath = '/data/workspace/myshixun/task4/'
    # 读取图像
    img = cv2.imread(filepath+'shanfeng.png')

    ########## Begin ##########

    # 1. 灰度化处理图像
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2. 高斯滤波
    blurred = cv2.GaussianBlur(grayImage, (5, 5), 0)
    # 3. 拉普拉斯算法
    dst = cv2.Laplacian(grayImage, cv2.CV_16S, ksize=3)
    
    # 4. 数据格式转换
    Laplacian = cv2.convertScaleAbs(dst)
    
    print(1)
    ########## End ##########
    # 保存图像
    cv2.imwrite(filepath + "out/laplacian.png",Laplacian)

def _log():
    filepath = '/data/workspace/myshixun/task5/'
   
    # 读取图像
    img = cv2.imread(filepath + 'shanfeng.png')
    ########## Begin ##########
    # 1. 灰度转换
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2. 边缘扩充处理图像并使用高斯滤波处理该图像
    image = cv2.copyMakeBorder(img, 2, 2, 2, 2, borderType=cv2.BORDER_REPLICATE)
    image = cv2.GaussianBlur(image, (3, 3), 0, 0)
    # 3. 使用Numpy定义LoG算子
    m1 = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]]) 
    # 4. 卷积运算
    # 为了使卷积对每个像素都进行运算，原图像的边缘像素要对准模板的中心。
    # 由于图像边缘扩大了2像素，因此要从位置2到行(列)-2
    r = image.shape[0]
    c = image.shape[1]
    image1 = np.zeros(image.shape)

    for k in range(0, 2): 
        for i in range(2, r - 2):
            for j in range(2, c - 2):
                image1[i, j] = np.sum((m1 * image[i - 2:i + 3, j - 2:j + 3, k]))

    # 5. 数据格式转换
    image1 = cv2.convertScaleAbs(image1)
    ########## End ##########

    cv2.imwrite(filepath + "out/log.png", image1)