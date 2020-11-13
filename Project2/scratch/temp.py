import cv2
# import pandas as pd
import numpy as np
import os
import glob

data_path = "face_images" 
res_path = "face_images\gray_scale"
img_dir = "Enter Directory of all images"
files = glob.glob(data_path+"\*.jpg")
data = []
for f1 in files:
    imageLab = cv2.cvtColor(cv2.imread(f1),cv2.COLOR_RGB2GRAY)
    print(f1.replace(data_path,res_path))
    cv2.imwrite(f1.replace(data_path,res_path),imageLab)
print("Done")
    
# img = cv2.imread(f1)
# image = cv2.imread("face_images/image00000.jpg")
# imageLab = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
# cv2.imshow('a',image)
# cv2.imshow('b',imageLab)
# cv2.waitKey(10000)
# print(temp.shape)