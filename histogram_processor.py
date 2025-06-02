import cv2
import numpy as np
import base64
import matplotlib.pyplot as plt
import io
# 在文件顶部导入部分添加
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

class HistogramProcessor:
    @staticmethod
    def process_image(image_file, histogram_type='gray'):
        try:
            if histogram_type.lower() == 'rgb':
                # 读取彩色图像
                img = HistogramProcessor._read_image(image_file, grayscale=False)
                
                # 计算RGB各通道直方图
                r_hist = cv2.calcHist([img], [2], None, [256], [0, 256])  # 红色通道
                g_hist = cv2.calcHist([img], [1], None, [256], [0, 256])  # 绿色通道
                b_hist = cv2.calcHist([img], [0], None, [256], [0, 256])  # 蓝色通道
                
                # 创建RGB直方图图表
                HistogramProcessor._create_histogram_plot(
                    [r_hist, g_hist, b_hist], 
                    ['red', 'green', 'blue'],
                    with_legend=True
                )
                
                # old achieve
                    # 转换原图为base64
                    # _, buffer = cv2.imencode('.jpg', img)
                    # img_str = base64.b64encode(buffer).decode()
                    # original_image = f"data:image/jpeg;base64,{img_str}"
            else:
                # 读取灰度图像
                img = HistogramProcessor._read_image(image_file)
                
                # 计算灰度直方图
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                
                # 绘制灰度直方图
                HistogramProcessor._create_histogram_plot(hist)
                
                # old achieve
                    # 转换原图为base64
                    # _, buffer = cv2.imencode('.jpg', img)
                    # img_str = base64.b64encode(buffer).decode()
                    # original_image = f"data:image/jpeg;base64,{img_str}"
            
            # 转换直方图为base64
            histogram_image = HistogramProcessor._plot_to_base64()

            # 转换原图为base64
            original_image = HistogramProcessor._image_to_base64(img)
            
            # old achieve
                # # 将matplotlib图表转换为图像
                # buf = io.BytesIO()
                # plt.savefig(buf, format='png')
                # buf.seek(0)
                
                # # 转换为base64
                # histogram_str = base64.b64encode(buf.read()).decode()
                # histogram_image = f"data:image/png;base64,{histogram_str}"
            
            return {
                "success": True,
                "original": original_image,
                "histogram": histogram_image
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def equalize_histogram(image_file):
        try:
            # 读取灰度图像
            gray_img = HistogramProcessor._read_image(image_file)
            
            # 灰度图均衡化
            equalized_img = cv2.equalizeHist(gray_img)
            
            # 计算原始和均衡化后的灰度直方图
            orig_hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
            equalized_hist = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])
            
            # 创建原始灰度直方图图表
            plt.figure(figsize=(10, 6))
            plt.plot(orig_hist, color='gray')
            plt.xlabel('pixel')
            plt.ylabel('frequency')
            plt.xlim([0, 256])
            
            # 将原始直方图图表转换为图像
            orig_hist_buf = io.BytesIO()
            plt.savefig(orig_hist_buf, format='png')
            orig_hist_buf.seek(0)
            orig_hist_str = base64.b64encode(orig_hist_buf.read()).decode()
            orig_hist_image = f"data:image/png;base64,{orig_hist_str}"
            plt.close()
            
            # 创建均衡化后的灰度直方图图表
            plt.figure(figsize=(10, 6))
            plt.plot(equalized_hist, color='gray')
            plt.xlabel('pixel')
            plt.ylabel('frequency')
            plt.xlim([0, 256])
            
            # 将均衡化后的直方图图表转换为图像
            equalized_hist_buf = io.BytesIO()
            plt.savefig(equalized_hist_buf, format='png')
            equalized_hist_buf.seek(0)
            equalized_hist_str = base64.b64encode(equalized_hist_buf.read()).decode()
            equalized_hist_image = f"data:image/png;base64,{equalized_hist_str}"
            plt.close()
            
            # 转换原始灰度图为base64
            _, orig_buffer = cv2.imencode('.jpg', gray_img)
            orig_img_str = base64.b64encode(orig_buffer).decode()
            original_image = f"data:image/jpeg;base64,{orig_img_str}"
            
            # 转换均衡化后的灰度图为base64
            _, equalized_buffer = cv2.imencode('.jpg', equalized_img)
            equalized_img_str = base64.b64encode(equalized_buffer).decode()
            equalized_image = f"data:image/jpeg;base64,{equalized_img_str}"
            
            return {
                "success": True,
                "original": original_image,
                "equalized": equalized_image,
                "original_histogram": orig_hist_image,
                "equalized_histogram": equalized_hist_image
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    # 在文件末尾添加以下方法
    
    @staticmethod
    def linear_transform_histogram(image_file):
        try:
            # 读取图片
            file_bytes = image_file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            
            # 读取灰度图像
            gray_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            # 创建输出图像
            h, w = gray_img.shape[:2]
            transformed_img = np.zeros(gray_img.shape, np.float64)
            
            # 通过遍历对不同像素范围内进行分段线性变化
            for i in range(h):
                for j in range(w):
                    pix = gray_img[i][j]
                    if pix < 70:
                        transformed_img[i][j] = 0.8 * pix
                    elif pix < 180:
                        transformed_img[i][j] = 2.6 * pix - 210
                    else:
                        transformed_img[i][j] = 0.138 * pix + 195
            
            # 四舍五入并转换为uint8类型
            transformed_img = np.around(transformed_img)
            transformed_img = transformed_img.astype(np.uint8)
            
            # 计算原始灰度直方图
            orig_hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
            
            # 计算变换后的灰度直方图
            transformed_hist = cv2.calcHist([transformed_img], [0], None, [256], [0, 256])
            
            # 创建原始灰度直方图图表
            plt.figure(figsize=(10, 6))
            plt.plot(orig_hist, color='gray')
            plt.xlabel('pixel')
            plt.ylabel('frequency')
            plt.xlim([0, 256])
            
            # 将原始直方图图表转换为图像
            orig_hist_buf = io.BytesIO()
            plt.savefig(orig_hist_buf, format='png')
            orig_hist_buf.seek(0)
            orig_hist_str = base64.b64encode(orig_hist_buf.read()).decode()
            orig_hist_image = f"data:image/png;base64,{orig_hist_str}"
            plt.close()
            
            # 创建变换后的灰度直方图图表
            plt.figure(figsize=(10, 6))
            plt.plot(transformed_hist, color='gray')
            plt.xlabel('pixel')
            plt.ylabel('frequency')
            plt.xlim([0, 256])
            
            # 将变换后的直方图图表转换为图像
            transformed_hist_buf = io.BytesIO()
            plt.savefig(transformed_hist_buf, format='png')
            transformed_hist_buf.seek(0)
            transformed_hist_str = base64.b64encode(transformed_hist_buf.read()).decode()
            transformed_hist_image = f"data:image/png;base64,{transformed_hist_str}"
            plt.close()
            
            # 转换原始灰度图为base64
            _, orig_buffer = cv2.imencode('.jpg', gray_img)
            orig_img_str = base64.b64encode(orig_buffer).decode()
            original_image = f"data:image/jpeg;base64,{orig_img_str}"
            
            # 转换变换后的灰度图为base64
            _, transformed_buffer = cv2.imencode('.jpg', transformed_img)
            transformed_img_str = base64.b64encode(transformed_buffer).decode()
            transformed_image = f"data:image/jpeg;base64,{transformed_img_str}"
            
            return {
                "success": True,
                "original": original_image,
                "transformed": transformed_image,
                "original_histogram": orig_hist_image,
                "transformed_histogram": transformed_hist_image
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
    def _create_histogram_plot(hist_data, color='gray', title=None, with_legend=False):
        """创建直方图图表"""
        plt.figure(figsize=(10, 6))
        
        if isinstance(hist_data, list) and isinstance(color, list):
            # 多通道直方图
            for i, (hist, clr, label) in enumerate(zip(hist_data, color, ['Red', 'Green', 'Blue'])):
                plt.plot(hist, color=clr, label=label)
            if with_legend:
                plt.legend()
        else:
            # 单通道直方图
            plt.plot(hist_data, color=color)
        
        plt.xlabel('pixel')
        plt.ylabel('frequency')
        plt.xlim([0, 256])
        if title:
            plt.title(title)
    
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