import argparse
#from asyncio.windows_events import NULL
from fileinput import filename
from pickletools import read_uint1
from flask import Flask, jsonify, request
from flask import make_response
import cv2
import pytesseract as tesseract
import os
import shutil
from PIL import Image
import pandas as pd
import numpy as np
from yolov5 import detect
import re

app = Flask(__name__)

def save_image(file):
    file.save('./temp/'+ file.filename)

def path_clear():
    if(os.path.exists('./temp')):
        shutil.rmtree('./temp/')

@app.route('/')
def web():
    return "flask test page"

@app.route('/predict', methods=['POST'])
def predict():    
    path_clear()
    
    if request.method == 'POST':
        os.makedirs('./temp')
        file = request.files['file']
        save_image(file) # 들어오는 이미지 저장

        path= './temp/'+file.filename
        img = Image.open(path)
        #img.info['dpi']

        size = (416,416)
        img = img.resize(size, Image.ANTIALIAS)
        dpi = (300, 300)
        img = img.rotate(-90)
        img.save(path, dpi=dpi)
        img.save('./blurtest/test.jpg', dpi=dpi)

        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='best.pt')
        parser.add_argument('--source', type=str, default='./temp/'+file.filename)
        parser.add_argument('--save-conf', default=True)
        parser.add_argument('--save-crop', default=True)
        parser.add_argument('--project', default='temp')
        parser.add_argument('--name', default='img')
        opt = parser.parse_args()

        detect.main(opt)
        
        img = Image.open('./temp/img/crops/letter/'+file.filename) # 사진 인식 못 했을 떄 예외 추가야해야 함.
        img = img.resize((260, 100), Image.ANTIALIAS)
        img.save('./temp/img/crops/letter/'+file.filename, dpi=dpi)
        img.save('./blurtest/test3.jpg', dpi=dpi)

        img = cv2.imread('./temp/img/crops/letter/'+file.filename)
        sharp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((2, 2), np.uint8)
        img4 = cv2.dilate(sharp, kernel, iterations=1)        
        cv2.imwrite('./blurtest/test4.jpg', img)
        img = cv2.erode(sharp, kernel, iterations=3)
        cv2.imwrite('./blurtest/test6.jpg', img)
        img = cv2.dilate(img, (2,2), iterations=1) 
        cv2.imwrite('./blurtest/test7.jpg', img)
        img = cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        cv2.imwrite('./temp/test.jpg', img)
        cv2.imwrite('./blurtest/test5.jpg', img)
        date = (tesseract.image_to_string(img))

        print(date)
        new_date = re.sub(r'[^0-9,.-]', '', date)
        print(new_date)
        new_date2 = pd.to_datetime(new_date)
        new_date2 = new_date2.date()
        print(new_date2)
        path_clear()

        res = {
            'date'   : new_date}

    return res

if __name__=="__main__":
    app.run(host="0.0.0.0", port = 8080, debug=True)