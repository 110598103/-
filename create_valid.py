# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 20:51:53 2021

@author: 張景堯
"""
from PIL import Image
import numpy as np
import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import random
import cv2
yourPath='train/'
allList = os.listdir(yourPath)
table=[]
for files in allList:
  yourpath=yourPath+files
  allfileList = os.listdir(yourpath)
  validPath='valid/'+files
  os.mkdir(validPath)
  print(len(allfileList))
  img_random=random.sample(allfileList,300)
  for i in img_random:
    img = cv2.imread(yourpath+'/'+i)
    cv2.imwrite(validPath+'/'+i,img)
    os.remove(yourpath+'/'+i)