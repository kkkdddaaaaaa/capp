import requests
import os
import json

def send_data(url):
    path_dir = 'C:/Users/lyfv/cap/cap_api_server/testimg'
    file_list = os.listdir(path_dir)
    for name in file_list:
        files = {
            'file':open('C:/Users/lyfv/cap/cap_api_server/testimg/' + name, 'rb')
        }
    res = requests.post(url,files=files)
    res_json = res.json()
        
    return res_json

url = "http://localhost:5000/predict"
print(send_data(url))