import cv2
import numpy as np
import base64
import matplotlib.pyplot as plt
import io
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import math


class SharpenProcessor:

    @staticmethod
    def frequency_domain_sharpening(image_file, filter='Ideal'):
        try:
            img = SharpenProcessor._read_image(image_file, grayscale=False)

            # 转换为灰度图像
            if len(img.shape) == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = img
                
            height, width = gray_img.shape

            # 执行傅里叶变换并中心化
            dft = cv2.dft(np.float32(gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)

            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
            cv2.normalize(magnitude_spectrum, magnitude_spectrum, 0, 255, cv2.NORM_MINMAX)
            magnitude_spectrum = np.uint8(magnitude_spectrum)

            lowPass = np.zeros((height, width), dtype=np.uint8)

            center_y = height // 2
            center_x = width // 2

            if filter.lower() == 'ideal':
                # 设置截止频率
                D0 = 40
    
                # 构建理想低通滤波器
                for i in range(height):
                    for j in range(width):
                        # 计算当前点到中心点的距离
                        y = i - center_y
                        x = j - center_x
                        D = np.sqrt(x**2 + y**2)
                        
                        # 距离小于截止频率时设为255（白色）
                        if D >= D0:
                            lowPass[i, j] = 255

            elif filter.lower() == 'butterworth':
                D0 = 20
                n = 2    # 滤波器阶数
    
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
                        lowPass[i, j] = np.uint8(H * 255)

            elif filter.lower() == 'gauss':
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
                        lowPass[i, j] = np.uint8(H * 255)
                
            else:
                raise ValueError(f"不支持的滤波器类型：{filter}，可选值为 'Ideal', 'ButterWorth', 'Gauss'。")

            # TODO 将滤波器应用到图像上
            # 应用滤波器
            lowPass_colored = np.stack([lowPass, lowPass], axis=-1)
            filtered_shift = dft_shift * lowPass_colored

            # 逆变换恢复图像
            f_ishift = np.fft.ifftshift(filtered_shift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

            # 归一化到0-255范围
            cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
            result = np.uint8(img_back)


            # 将结果图转为 Base64
            filter_base64 = SharpenProcessor._image_to_base64(lowPass)
            frequency_base64 = SharpenProcessor._image_to_base64(magnitude_spectrum)
            result_base64 = SharpenProcessor._image_to_base64(result)

            return {
                "success": True,
                "filterImage": filter_base64,
                "frequencyImage": frequency_base64,
                "resultImage": result_base64
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    
    @staticmethod
    def spatial_domain_sharpening(image_file, operator='Roberts'):
        try:
            img = SharpenProcessor._read_image(image_file)

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


    @staticmethod
    def _read_image(image_file, grayscale=True):
        """读取图像文件并返回图像数组"""
        file_bytes = image_file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        if grayscale:
            return cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        else:
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    @staticmethod
    def _image_to_base64(img):
        """将图像转换为base64编码"""
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode()
        return f"data:image/jpeg;base64,{img_str}"