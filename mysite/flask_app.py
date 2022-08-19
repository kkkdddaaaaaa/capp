import argparse
from asyncio.windows_events import NULL
from fileinput import filename
from pickletools import read_uint1
from flask import Flask, jsonify, request
from flask import make_response
import cv2
import pytesseract as tesseract
import os
import shutil
from yolov5 import detect

app = Flask(__name__)

# POST 통신으로 들어오는 이미지를 저장하고 모델로 추론하는 과정

def save_image(file):
    file.save('./temp/'+ file.filename)

@app.route('/')
def web():
    return "Lungnaha's flask test page"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        save_image(file) # 들어오는 이미지 저장

        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model path(s)')
        parser.add_argument('--source', type=str, default='./temp/'+file.filename, help='file/dir/URL/glob, 0 for webcam')
        parser.add_argument('--save-conf', default=True, help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', default=True, help='save cropped prediction boxes')
        parser.add_argument('--project', default='temp', help='save results to project/name')
        parser.add_argument('--name', default='img', help='save results to project/name')
        opt = parser.parse_args()

        detect.main(opt)
        
        img = cv2.imread('./temp/img/crops/letter/'+file.filename)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sharp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(sharp, 127, 255, cv2.THRESH_BINARY)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        open = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_open)
        
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,7))
        dilate = cv2.erode(open,kernel_dilate)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        close = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel_close)
        
        cv2.imwrite('./testimg/test.jpg', close)
        res = (tesseract.image_to_string(close))
        print(res)

        os.remove('./temp/'+file.filename)
        os.remove('./testimg/test.jpg')
        shutil.rmtree('./temp/img')

        

        year = {
            'year'   : res
        }

    return year

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)