import cv2
import matplotlib.pyplot as plt


def show_color_histogram(img_path):
    # 检查图片是否成功读取    
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # 检查图片是否成功读取
    if img is None:
        print("Error: Could not read the image.")
        exit()

    r_hist = cv2.calcHist([img], [2], None, [256], [0, 256])  # 红色通道
    g_hist = cv2.calcHist([img], [1], None, [256], [0, 256])  # 绿色通道
    b_hist = cv2.calcHist([img], [0], None, [256], [0, 256])  # 蓝色通道

    plt.figure()

    # 绘制各通道直方图曲线
    plt.plot(r_hist, color='red', label='Red')
    plt.plot(g_hist, color='green', label='Green')
    plt.plot(b_hist, color='blue', label='Blue')

    # 设置坐标轴范围
    plt.xlim([0, 256])

    plt.show()

def show_gray_histogram(img_path):
    # 检查图片是否成功读取
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 检查图片是否成功读取
    if img is None:
        print("Error: Could not read the image.")
        exit()

    # 计算灰度直方图
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # 绘制灰度直方图
    plt.figure()
    
    plt.plot(hist)
    plt.xlim([0, 256])

    plt.show()



