from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from resize_processor import ResizeProcessor
from colorspace_processor import ColorSpaceProcessor
from arithmetic_processor import ArithmeticProcessor
from logtrans_processor import LogTransProcessor
from histogram_processor import HistogramProcessor
from segmentation_processor import SegmentationProcessor
from smooth_processor import SmoothProcessor
from sharpen_processor import SharpenProcessor

app = Flask(__name__)
CORS(app)

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/resize', methods=['GET'])
def resize_page():
    return render_template('resize.html')

@app.route('/resize', methods=['POST'])
def resize_image():
    if 'image' not in request.files:
        return jsonify({"error": "没有上传图片"}), 400
    
    try:
        # 获取所有参数
        width = int(request.form.get('width', 800))
        height = int(request.form.get('height', 600))
        tx = int(request.form.get('tx', 0))  # 水平平移
        ty = int(request.form.get('ty', 0))  # 垂直平移
        angle = float(request.form.get('angle', 0))  # 旋转角度
        
        # 调用处理函数
        result = ResizeProcessor.process_image(
            request.files['image'],
            width,
            height,
            tx=tx,
            ty=ty,
            angle=angle
        )
        
        if result["success"]:
            return jsonify(result)
        else:
            return jsonify({"error": result["error"]}), 400
    except ValueError as e:
        return jsonify({"error": "参数值无效"}), 400

@app.route('/colorspace', methods=['GET'])
def colorspace_page():
    return render_template('colorspace.html')

@app.route('/analyze_colorspace', methods=['POST'])
def analyze_colorspace():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    analysis_type = request.form.get('type', 'rgb')
    result = ColorSpaceProcessor.process_image(request.files['image'], analysis_type)
    return jsonify(result)

@app.route('/arithmetic')
def arithmetic():
    return render_template('arithmetic.html')

@app.route('/arithmetic', methods=['POST'])
def process_arithmetic():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "请上传两张图片"}), 400
    
    operation = request.form.get('operation', 'add')
    result = ArithmeticProcessor.process_images(
        request.files['image1'],
        request.files['image2'],
        operation
    )
    
    return jsonify(result)

@app.route('/logtrans')
def logtrans():
    return render_template('logtrans.html')

@app.route('/logtrans', methods=['POST'])
def process_logtrans():
    if 'image' not in request.files:
        return jsonify({"error": "请上传图片"}), 400
    
    v = request.form.get('v', '100')
    result = LogTransProcessor.process_image(
        request.files['image'],
        v
    )
    
    return jsonify(result)

# 在现有导入语句下添加
from histogram_processor import HistogramProcessor

@app.route('/histogram')
def histogram():
    return render_template('histogram.html')

@app.route('/generate_histogram', methods=['POST'])
def generate_histogram():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    histogram_type = request.form.get('type', 'gray')
    result = HistogramProcessor.process_image(request.files['image'], histogram_type)
    return jsonify(result)

@app.route('/equalize_histogram', methods=['POST'])
def equalize_histogram():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    result = HistogramProcessor.equalize_histogram(request.files['image'])
    return jsonify(result)

@app.route('/linear_transform_histogram', methods=['POST'])
def linear_transform_histogram():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    result = HistogramProcessor.linear_transform_histogram(request.files['image'])
    return jsonify(result)


@app.route('/histogram_normalize')
def histogram_normalize():
    return render_template('histogram_normalize.html')


@app.route('/normalize_histogram', methods=['POST'])
def normalize_histogram():
    if 'source_image' not in request.files or 'target_image' not in request.files:
        return jsonify({"success": False, "error": "请上传原图像和目标图像"})
    
    result = HistogramProcessor.normalize_histogram(
        request.files['source_image'],
        request.files['target_image']
    )
    return jsonify(result)


@app.route('/segmentation')
def segmentation():
    return render_template('segmentation.html')

@app.route('/basic_enhance_detail', methods=['POST'])
def basic_enhance_detail():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    result = SegmentationProcessor.basic_enhance_detail(request.files['image'])
    return jsonify(result)

@app.route('/edge_detection', methods=['POST'])
def edge_detection():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    operator = request.form.get('operator', 'roberts')
    result = SegmentationProcessor.edge_detection(request.files['image'], operator)
    return jsonify(result)

@app.route('/line_change_detection', methods=['POST'])
def line_change_detection():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    result = SegmentationProcessor.line_change_detection(request.files['image'])
    return jsonify(result)


@app.route('/smooth')
def smooth():
    return render_template('smooth.html')

@app.route('/frequency_smoothing', methods=['POST'])
def frequency_smoothing():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    filter = request.form.get('filter', 'Ideal')
    result = SmoothProcessor.frequency_domain_smoothing(request.files['image'], filter)
    return jsonify(result)

@app.route('/spatial_smoothing', methods=['POST'])
def spatial_smoothing():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    filter = request.form.get('filter', 'Mean')
    result = SmoothProcessor.spatial_domain_smoothing(request.files['image'], filter)
    return jsonify(result)


@app.route('/sharpen')
def sharpen():
    return render_template('sharpen.html')

@app.route('/frequency_sharpening', methods=['POST'])
def frequency_sharpening():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    filter = request.form.get('filter', 'Ideal')
    result = SharpenProcessor.frequency_domain_sharpening(request.files['image'], filter)
    return jsonify(result)

@app.route('/spatial_sharpening', methods=['POST'])
def spatial_sharpening():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "没有上传图片"})
    
    operator = request.form.get('operator', 'Roberts')
    result = SharpenProcessor.spatial_domain_sharpening(request.files['image'], operator)
    return jsonify(result)



if __name__ == '__main__':
    app.run(debug=True)