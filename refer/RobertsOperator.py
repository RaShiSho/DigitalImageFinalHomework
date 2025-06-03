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

    print(1)
    ########## End ##########
    
    # 保存图像
    cv2.imwrite(filepath+"out/roberts.png",Roberts)
