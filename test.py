# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 20:58:38 2021

@author: 張景堯
"""
from keras.models import load_model
#載入model
print('載入model')
model=load_model('model_128.hdf5')
print('---------------------------------------------')
import os
import cv2
#建立人物名稱
print('建立人物名稱')
yourPath='theSimpsons-train/train/'
allList = os.listdir(yourPath)
name={}
count=0
for files in allList:
    name.update({count:files})
    count=count+1
print(name)
print('---------------------------------------------')

import numpy as np
#讀取圖片，並壓所到128*128
def read_image(i):
    image=cv2.resize(cv2.imread('theSimpsons-test/test/'+str(i+1)+'.jpg'),(128,128))
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    images=[]
    images.append(image)
    images=np.array(images,dtype=np.float32)/255
    return images
#寫入CSV
with open('test_data128.csv','w') as f:
        f.write('id,character\n')
f.close()
with open('test_data128.csv','a') as f:
    for i in range(10791):
        read_test=read_image(i)
        read_test.shape
        y_predict=model.predict(read_test)
        classes_x=np.argmax(y_predict,axis=1)
        print("寫入:",i)
        f.write(str(i+1)+','+name[classes_x[0]]+'\n')