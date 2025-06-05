import cv2
import numpy as np

def Ideal_LowPassFilter(im):
    # 获取图像尺寸（处理单通道或彩色图像）
    height, width = im.shape[0], im.shape[1]
    
    # 创建单通道滤波器模板（全黑）
    Ideal_LowPass = np.zeros((height, width), dtype=np.uint8)
    
    # 计算中心点坐标
    center_y = height // 2
    center_x = width // 2
    
    # 设置截止频率
    D0 = 20
    
    # 构建理想低通滤波器
    for i in range(height):
        for j in range(width):
            # 计算当前点到中心点的距离
            y = i - center_y
            x = j - center_x
            D = np.sqrt(x**2 + y**2)
            
            # 距离小于截止频率时设为255（白色）
            if D <= D0:
                Ideal_LowPass[i, j] = 255
                
    return Ideal_LowPass

def ButterWorth_LowPassFilter(im):
    height, width = im.shape[0], im.shape[1]
    
    # 创建滤波器模板（单通道）
    ButterWorth_LowPass = np.zeros((height, width), dtype=np.uint8)
    
    # 计算中心点坐标
    center_y = height // 2
    center_x = width // 2
    
    # 巴特沃斯参数
    D0 = 20  # 截止频率
    n = 2    # 滤波器阶数
    
    # 构建巴特沃斯低通滤波器
    for i in range(height):
        for j in range(width):
            # 计算当前点到中心点的距离
            y = i - center_y
            x = j - center_x
            D = np.sqrt(x**2 + y**2)
            
            # 计算巴特沃斯传递函数值
            if D == 0:  # 避免除以零
                H = 1.0
            else:
                # 核心公式：H = 1 / [1 + (D/D0)^(2n)]
                H = 1.0 / (1.0 + (D / D0) ** (2 * n))
            
            # 将0-1范围的值映射到0-255范围
            ButterWorth_LowPass[i, j] = np.uint8(H * 255)
    
    return ButterWorth_LowPass

def Gauss_LowPassFilter(im):
    height, width = im.shape[0], im.shape[1]
    
    # 创建滤波器模板（单通道）
    Gauss_LowPass = np.zeros((height, width), dtype=np.uint8)
    
    # 计算中心点坐标
    center_y = height // 2
    center_x = width // 2
    
    # 高斯参数
    D0 = 20  # 截止频率
    
    # 构建高斯低通滤波器
    for i in range(height):
        for j in range(width):
            # 计算当前点到中心点的距离
            y = i - center_y
            x = j - center_x
            D = math.sqrt(x**2 + y**2)
            
            # 计算高斯传递函数值
            # 公式：H = e^(-(D²)/(2*D0²))
            exponent = -1.0 * (D**2) / (2.0 * D0**2)
            H = math.exp(exponent)
            
            # 将0-1范围的值映射到0-255范围
            Gauss_LowPass[i, j] = np.uint8(H * 255)
    
    return Gauss_LowPass

# 使用示例
if __name__ == "__main__":
    # 读取输入图像（可处理灰度或彩色）
    input_image = cv2.imread("input.jpg", cv2.IMREAD_ANYCOLOR)
    
    # 生成滤波器
    filter_mask = Ideal_LowPassFilter(input_image)
    
    # 保存结果
    cv2.imwrite("ideal_lowpass_filter.jpg", filter_mask)