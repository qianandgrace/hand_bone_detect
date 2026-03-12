from flask import Flask,render_template,request,send_file
from PIL import Image
import io
import cv2
from detect_utils import load_model,detect_img
import base64

app = Flask(__name__)
yolov5_model,cls_models = load_model()

@app.route("/")
def index():
    return render_template("client.html")

@app.route("/predict",methods=['POST'])
def predict():
    if request.method =='POST':
        file = request.files['file']
        img_bytes = file.read()
        #把字节数据转成图片
        img = Image.open(io.BytesIO(img_bytes))
        img.save("test_data/example.jpg")
        img_path="test_data/example.jpg"
        sex = request.form['sex']
        export = detect_img(yolov5_model,cls_models,img_path,sex)
        print(export)
        with open('detect_result/detect.jpg', 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        # 将 Base64 编码的图片转换为 HTML 可以识别的 URL 格式
        image_url = f"data:image/jpeg;base64,{encoded_string}"
        # 渲染模板并传递文本和图片 URL
        return render_template('result.html', text=f'{export}', image_url=image_url)
    else:
        return ""

if __name__== "__main__":
    app.run(host="0.0.0.0",port=5000)