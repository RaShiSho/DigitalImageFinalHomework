import cv2
import numpy as np

def task1():

    filename = '/data/workspace/myshixun/task1/CRH.png'
    
    ########## Begin ##########
    # 1. 灰度模式读取图像，图像名为CRH
    CRH = cv2.imread(filename, 0)
    # 2. 计算图像梯度。首先要对读取的图像进行数据变换，因为使用了
    # numpy对梯度进行数值计算，所以要使用
    # CRH.astype('float')进行数据格式变换。
    CRH = CRH.astype('float')
    row, col = CRH.shape
    gradient = np.zeros((row, col))
    # 3. 对图像进行增强，增强后的图像变量名为sharp
    for x in range(row - 1):
        for y in range(col - 1):
            gx = abs(CRH[x + 1, y] - CRH[x, y])
            gy = abs(CRH[x, y + 1] - CRH[x, y])
            gradient[x, y] = gx + gy
    sharp = CRH + gradient

    print(1)
    ########## End ##########
    
    sharp = np.where(sharp > 255, 255, sharp)
    sharp = np.where(sharp < 0, 0, sharp)
    
    # 数据类型变换
    gradient = gradient.astype('uint8')
    sharp = sharp.astype('uint8')
    
    # 保存图像
    filepath = '/data/workspace/myshixun/task1/'
    cv2.imwrite(filepath + 'out/gradient.png', gradient)