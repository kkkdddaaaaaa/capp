import argparse
import pytesseract as tesseract
import pandas as pd
import numpy as np
import os
import re
import shutil
import cv2
from asyncio.windows_events import NULL
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS
from fileinput import filename
from pickletools import read_uint1
from flask import Flask, jsonify, request, make_response
from datetime import datetime
from yolov5 import detect

app = Flask(__name__)


def save_image(file):                                 # 사진 저장
    file.save('./temp/' + file.filename)

def path_clear():                                     # 경로 정리
    if (os.path.exists('./temp')):
        shutil.rmtree('./temp/')

@app.route('/')
def web():
    return "flask test page"


@app.route('/predict', methods=['POST'])
def predict():
    path_clear()                                      # 임시파일 제거

    if request.method == 'POST':
        os.makedirs('./temp')
        file = request.files['file']                  # 들어오는 이미지 저장                                                      
        save_image(file)
        #path = './temp/'+file.filename       

        tempimg = cv2.imread('./temp/'+file.filename)                                           # 로컬 폴더에 이미지 로그 저장
        c = datetime.now()
        c = c.strftime("%Y-%m-%d (%H %M %S)")
        cv2.imwrite('./imagelog/'+ c +'.jpg', tempimg)

        # size = (416,416)                                                                       # Tesseract를 사용하기 위한 이미지 전처리 1
        # img = img.resize(size, Image.ANTIALIAS)                                                # 이미지 리사이즈 및 DPI 설정
        dpi = (300, 300)
        
        #img = img.rotate(-90)
        #img.save(path, dpi=dpi)
                                                                                                

        parser = argparse.ArgumentParser()                                                       # yolov5의 detect.py를 사용하기 위한 인자 설정
        parser.add_argument('--weights', nargs='+', type=str, default='best.pt')
        parser.add_argument('--source', type=str, default='./temp/'+file.filename)
        parser.add_argument('--save-conf', default=True)
        parser.add_argument('--save-crop', default=True)
        parser.add_argument('--project', default='temp')
        parser.add_argument('--name', default='img')
        opt = parser.parse_args()

        detect.main(opt)

        try:            
            img = Image.open('./temp/img/crops/letter/'+file.filename)                           # 사진 인식 못 했을 떄 예외처리.
        except FileNotFoundError as e:
            print(e)
            return ""

        # Tesseract를 사용하기 위한 이미지 전처리 2
        #img = cv2.imread('./temp/img/crops/letter/'+file.filename)
        #cv2.imwrite('./temp/img/crops/letter/'+file.filename, img)
       
        img = img.resize((260, 100), Image.ANTIALIAS)                                           # yolov5로 crop한 이미지 리사이즈 후 저장                                     
        img.save('./temp/img/crops/letter/'+file.filename, dpi=dpi)
        img = cv2.imread('./temp/img/crops/letter/'+file.filename)

        '''img2 = cv2.imread('./temp/img/crops/letter/' + file.filename, cv2.IMREAD_COLOR)      # 배경 제거하는 코드
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        bgr = img2[:, :, :]
        mask = cv2.inRange(bgr, (120, 120, 120), (255, 255, 255))
        bgr_new = bgr.copy()
        bgr_new[mask > 0] = (255, 255, 255)
        cv2.imwrite('./blurtest/bgr_new.jpg', bgr_new)'''

        
        sharp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                           # 그레이스케일        
        kernel = np.ones((1, 1), np.uint8)                                                      # Tesseract를 사용하기 위한 이미지 전처리 3        
        img = cv2.dilate(sharp, (1, 1), iterations=1)                                           # 이미지 Closing, Opening
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.threshold(cv2.bilateralFilter(
            img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]                    # 이진 필터

        cv2.imwrite('./temp/test.jpg', img)
        cv2.imwrite('./blurtest/test5.jpg', img)
        img = Image.open('./temp/test.jpg')
        img.save('./temp/test.jpg', dpi=dpi)
        img = cv2.imread('./temp/test.jpg')
        
        date = (tesseract.image_to_string(img))                                                 # Tesseract-OCR을 통한 이미지 문자화


        date = re.sub(r'[^0-9,.-]', '', date)  
        arr = re.split("[.,-]", date)       # 추가해야 할 것 : month에 한자리 숫자가 올 경우, 앞에 0을 추가하는 조건문
                                            # 8자리 끊어서 연도계산 하게 하기 
        try: #연도가 없는 유통기한 분류
            if(arr[2]==NULL):
                print(arr[1])
        except IndexError as eeeee:
            year = datetime.now().year
            new_date = str(year) +'-'+ str(arr[0]) +'-'+ str(arr[1])
            print(new_date)
            arr = re.split("[.,-]", new_date)

        if(len(arr[0])==2): # 연도가 4자리가 아닌 유통기한 분류
            arr[0] = '20' + arr[0]
        
        #if(len(arr[1])==1):
        #    arr[1] = '0' + arr[1]

        new_date = ''
        for i in arr:
            new_date += i +'.'

        print(arr)
        print(new_date)

        date = re.sub(r'[^0-9]', '', new_date)                                                      # 숫자와 콤마, 하이픈만 추출                                                
        date = date[0:8]        
        date = pd.to_datetime((date), yearfirst=True)                                           # 문자열을 datetime 형식으로 변경
        date = date.strftime("%Y-%m-%d")
        print(date)
        
        path_clear()                                                                            # 임시파일 제거

        res = {
            'date': date}

    return res


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
