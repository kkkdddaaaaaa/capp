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
from yolov5 import detect

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

        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='best.pt')
        parser.add_argument('--source', type=str, default='./temp/'+file.filename)
        parser.add_argument('--save-conf', default=True)
        parser.add_argument('--save-crop', default=True)
        parser.add_argument('--project', default='temp')
        parser.add_argument('--name', default='img')
        opt = parser.parse_args()

        detect.main(opt)
        
        img = cv2.imread('./temp/img/crops/letter/'+file.filename)

        sharp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(sharp, 127, 255, cv2.THRESH_BINARY)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        open = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_open)
        
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,7))
        dilate = cv2.erode(open,kernel_dilate)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        close = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel_close)
        
        cv2.imwrite('./temp/test.jpg', close)
        date = (tesseract.image_to_string(close))

        print(date)
        path_clear()

        res = {
            'date'   : date
        }

    return res

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)