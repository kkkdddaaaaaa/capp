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

def save_image(file):                                 # 사진 저장
    file.save('./temp/'+ file.filename)

def path_clear():                                     # 경로 정리
    if(os.path.exists('./temp')):
        shutil.rmtree('./temp/')

@app.route('/')
def web():
    return "flask test page"

@app.route('/predict', methods=['POST'])        
def predict():    
    path_clear()                                      # 임시파일 제거
    
    if request.method == 'POST':
        os.makedirs('./temp')
        file = request.files['file']
        save_image(file)                                                                       # 들어오는 이미지 저장

        path= './temp/'+file.filename
        img = Image.open(path)                                                                 # 로컬에 저장한 이미지 열기

        size = (416,416)                                                                       # Tesseract를 사용하기 위한 이미지 전처리 1
        img = img.resize(size, Image.ANTIALIAS)                                                # 이미지 리사이즈 및 DPI 설정
        dpi = (300, 300)
        img = img.rotate(-90)
        img.save(path, dpi=dpi)
        img.save('./blurtest/test.jpg', dpi=dpi)

        parser = argparse.ArgumentParser()                                                     # yolov5의 detect.py를 사용하기 위한 인자 설정
        parser.add_argument('--weights', nargs='+', type=str, default='best.pt')
        parser.add_argument('--source', type=str, default='./temp/'+file.filename)
        parser.add_argument('--save-conf', default=True)
        parser.add_argument('--save-crop', default=True)
        parser.add_argument('--project', default='temp')
        parser.add_argument('--name', default='img')
        opt = parser.parse_args()

        detect.main(opt)
        try:
            img = Image.open('./temp/img/crops/letter/'+file.filename)                         # 사진 인식 못 했을 떄 예외처리.
        except FileNotFoundError as e:
            print(e)
            return 

        img = img.resize((260, 100), Image.ANTIALIAS)                                          # Tesseract를 사용하기 위한 이미지 전처리 2
        img.save('./temp/img/crops/letter/'+file.filename, dpi=dpi)                            # yolov5로 crop한 이미지 리사이즈 후 저장
        
        img = cv2.imread('./temp/img/crops/letter/'+file.filename)
        sharp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                          # 그레이스케일
        kernel = np.ones((2, 2), np.uint8)                                                     # Tesseract를 사용하기 위한 이미지 전처리 3
        img = cv2.dilate(sharp, (1, 1), iterations=1)                                          # 이미지 Closing, Opening
        img = cv2.erode(img, kernel, iterations=3)
        img = cv2.dilate(img, kernel, iterations=1) 
        img = cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]    # 이진 필터

        cv2.imwrite('./temp/test.jpg', img)
        cv2.imwrite('./blurtest/test5.jpg', img)
        date = (tesseract.image_to_string(img))                                                # Tesseract-OCR을 통한 이미지 문자화

        print(date)
        date = re.sub(r'[^0-9,.-]', '', date)                                                  # 숫자와 콤마, 하이픈만 추출
        date = pd.to_datetime((date), yearfirst = True)                                        # 문자열을 datetime 형식으로 변경
        date = date.strftime("%Y-%m-%d")
        print(date)
        path_clear()                                                                           # 임시파일 제거

        res = {
            'date'   : date}

    return res

if __name__=="__main__":
    app.run(host="0.0.0.0", port = 6000, debug=True)