# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 14:19:39 2020

@author: sijas
"""

# The I/O module is used for importing the image
import cv2
import os
#import matplotlib.pyplot as plt

def denoise(path):
    images_list = []
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            images_list.append(file)
    print(len(images_list))
    
    for filename in images_list:
        img = cv2.imread(str(path) + str(filename), 0)
        print(str(path) + str(filename))
        img = cv2.medianBlur(img,5)
        ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        
        cv2.imwrite(str(path) + str(filename), th1)

denoise(r"C:\Users\sijas\.tensorflow\workspace\hack_demo\images\test\\")