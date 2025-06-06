import cv2
import numpy as np
import base64
import matplotlib.pyplot as plt
import io
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import math
import random


class RestoreProcessor:

    # 椒盐噪声的阈值
    saltpepper_prob = 0.2
    saltpepper_thres = 1 - saltpepper_prob

    @staticmethod
    def add_noise(image_file, type="saltpepper"):
        try:
            img = RestoreProcessor._read_image(image_file, grayscale=False)

            if type.lower() == 'saltpepper':
                # 待输出的图片
                output = np.zeros(img.shape, np.uint8)
                
                # 遍历图像，获取叠加噪声后的图像
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        rdn = random.random()
                        if rdn < RestoreProcessor.saltpepper_prob:
                            # 添加胡椒噪声
                            output[i][j] = 0
                        elif rdn > RestoreProcessor.saltpepper_thres:
                            # 添加食盐噪声
                            output[i][j] = 255
                        else:
                            # 不添加噪声
                            output[i][j] = img[i][j]
                result = output

            elif type.lower() == 'gauss':
                # 将图片的像素值归一化，存入矩阵中
                image = np.array(img / 255, dtype=float)
                # 生成正态分布的噪声，其中0表示均值，0.1表示方差
                noise = np.random.normal(0, 0.1, image.shape)
                # 将噪声叠加到图片上
                out = image + noise
                # 将图像的归一化像素值控制在0和1之间，防止噪声越界
                out = np.clip(out, 0.0, 1.0)
                # 将图像的像素值恢复到0到255之间
                result = np.uint8(out*255)
                
            else:
                raise ValueError(f"不支持的处理类型：{type}，可选值为 'erosion', '', 'Gauss'。")

        
            # 将结果图转为 Base64
            result_base64 = RestoreProcessor._image_to_base64(result)

            return {
                "success": True,
                "resultImage": result_base64
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


    @staticmethod
    def meanFiltering(image_file, filter_x=3, filter_y=3):
        try:
            img = RestoreProcessor._read_image(image_file, grayscale=False)

            # 均值滤波
            result = cv2.blur(img, (int(filter_x), int(filter_y)))
        
            # 将结果图转为 Base64
            result_base64 = RestoreProcessor._image_to_base64(result)

            return {
                "success": True,
                "resultImage": result_base64
            }

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
            return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    @staticmethod
    def _image_to_base64(img):
        """将图像转换为base64编码"""
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode()
        return f"data:image/jpeg;base64,{img_str}"