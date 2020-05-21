import csv
import cv2
import numpy as np
import os
import imutils
import skimage
from scipy import ndarray
from skimage import transform
from skimage import util
import random
from matplotlib import pyplot
from sklearn.cluster import KMeans
from random import randint
import copy

with open("/home/vanjani/Desktop/image/training_aug.csv","w") as outfile:
                    writer=csv.writer(outfile)

flag=0
def addtocsv(count,output):
    row={}
    row["image"]="image_"+str(count)
    row["output"]=output
    
    keys=[]
    values=[]
    global flag
    for key,value in row.items():
        keys.append(key)
        values.append(value)
    with open("/home/vanjani/Desktop/image/training_aug.csv","a") as outfile:
        writer=csv.writer(outfile)
        if flag==0:
            writer.writerow(keys)
            flag=1
        writer.writerow(values)


count=0
train_path = "/home/vanjani/Desktop/image/processeddataset_aug/corrected_melanoma/"
train_names = os.listdir(train_path)

for file in train_names:
    path = "/home/vanjani/Desktop/image/processed_aug/"
    file_name = train_path+file
    image = cv2.imread(file_name)
    cv2.imwrite(os.path.join(path,"image_" + str(count) + ".jpg"), image)
    addtocsv(count,0)
    count+=1

train_path = "/home/vanjani/Desktop/image/processeddataset_aug/corrected_non_melanoma/"
train_names = os.listdir(train_path)

for file in train_names:
    path = "/home/vanjani/Desktop/image/processed_aug/"
    file_name = train_path+file
    image = cv2.imread(file_name)
    cv2.imwrite(os.path.join(path,"image_" + str(count) + ".jpg"), image)
    addtocsv(count,1)
    count+=1

