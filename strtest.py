from asyncio.windows_events import NULL
import re
import pandas as pd
from datetime import datetime

str1 = "22.7.21"
str2 = "23.07.23"
str3 = "2023-05-265"
str4 = "07.24"
str5 = "22,12,30"

date = str1

date = re.sub(r'[^0-9,.-]', '', date)  
arr = re.split("[.,-]", date)

# 추가해야 할 것 : month에 한자리 숫자가 올 경우, 앞에 0을 추가하는 조건문
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
    arr[0] = '20'+arr[0]

if(len(arr[1])==1):
    arr[1] = '0'+arr[1]


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