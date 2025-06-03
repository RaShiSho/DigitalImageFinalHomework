import cv2
import numpy as np
import base64
import matplotlib.pyplot as plt
import io
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

class SegmentationProcessor:
    @staticmethod
    def basic_enhance_detail(image_file):
        try:
            # 读取灰度图像
            gray_img = SegmentationProcessor._read_image(image_file)
            
            gray_img = gray_img.astype('float')
            row, col = gray_img.shape
            gradient = np.zeros((row, col))

            for x in range(row - 1):
                for y in range(col - 1):
                    gx = abs(gray_img[x + 1, y] - gray_img[x, y])
                    gy = abs(gray_img[x, y + 1] - gray_img[x, y])
                    gradient[x, y] = gx + gy
            sharp = gray_img + gradient

            sharp = np.where(sharp > 255, 255, sharp)
            sharp = np.where(sharp < 0, 0, sharp)
            
            # 转换原始和均衡化后的灰度图为base64
            gray_image = SegmentationProcessor._image_to_base64(gray_img)
            gradient_image = SegmentationProcessor._image_to_base64(gradient)
            enhanced_image = SegmentationProcessor._image_to_base64(sharp)
            
            return {
                "success": True,
                "originalGrayImage": gray_image,
                "gradientImage": gradient_image,
                "enhancedGrayImage": enhanced_image
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
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    @staticmethod
    def _plot_to_base64():
        """将当前图表转换为base64编码的图像"""
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        return f"data:image/png;base64,{img_str}"

    @staticmethod
    def _image_to_base64(img):
        """将图像转换为base64编码"""
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode()
        return f"data:image/jpeg;base64,{img_str}"