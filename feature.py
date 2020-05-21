#extracts 52+52+59+64=227 features of an image
import csv
import cv2
import numpy as np
import os
import glob
import mahotas as mt
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import normalize
import cvutils
from skimage import feature
from scipy.stats import kurtosis, skew
from sklearn import svm
import math

#Extracts 59 LBP Features
class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius
 
    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="default")
        
        uniform = [0,1,2,3,4,6,7,8,12,14,15,16,24,28,30,31,32,48,56,60,62,63,64,96,112,120,124,126,127,128,129,131,135,143,159,191,192,193,195,199,207,223,224,225,227,231,239,240,241,243,247,248,249,251,252,253,254,255]
        dis = {}
        for i in range(len(uniform)):
            dis[uniform[i]]=i
        hs=[]
        for i in range(59):
            hs.append(0)
        non_uniform_count=0
        for i in range(256):
            for j in range(256):
                if lbp[i][j] in dis:    
                    hs[dis[lbp[i][j]]]+=1
                else:
                    non_uniform_count+=1
        final_hs=[]
        for i in range(58):
            c=hs[i]
            for j in range(c):
                final_hs.append(i)
        for i in range(non_uniform_count):
            final_hs.append(58)
        hs = final_hs
        hs=np.array(hs)
        (hs,_) = np.histogram(hs, bins=59, weights=np.ones(len(hs)) / len(hs))
        return hs

desc = LocalBinaryPatterns(8,1)
	
# load the training dataset
train_path = "/home/vanjani/Desktop/Melanoma/processed_aug"
train_names = os.listdir(train_path)

#To extract GLCM based features
#Extracts 52 features
def extract_features(image):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(image)

        ht=[];
        for i in range(4):
                for j in range(len(textures[i])):
                        ht.append(textures[i][j])
        '''
        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean
        '''
        return ht

#To extract color features like (Mean, standard deviation, skewness and kurtosis) from image
#Extracts 52 features
def extract_color(image, number_of_channels, mask):
    fi=[]
    N=len(mask)
    if number_of_channels==1:
        for i in range(0,1):
            temp=[]
            fi.append(temp)

        for (i,j) in mask:
            fi[0].append(image[i][j])

    else:
        for i in range(0,3):
            temp=[]
            fi.append(temp)
            
        for (i,j) in mask:
            fi[0].append(image[i][j][0])
            fi[1].append(image[i][j][1])
            fi[2].append(image[i][j][2])

    fi=np.array(fi)

    #Mean
    x_mean=[]
    x_stddev=[]
    x_skewness=[]
    x_kurtosis=[]
    for i in range(0,number_of_channels):
        x_mean.append(np.sum(fi[i])/N)
        x_stddev.append(math.sqrt(np.sum(np.power([x - x_mean[i] for x in fi[i]],2))/N))
        x_skewness.append(np.sum(np.power(np.true_divide([x - x_mean[i] for x in fi[i]],x_stddev[i]),3))/N)
        x_kurtosis.append(np.sum(np.power(np.true_divide([x - x_mean[i] for x in fi[i]],x_stddev[i]),4))/N)
    
    feature_vector=[]
    for i in range(len(x_mean)):
        feature_vector.append(x_mean[i])
    for i in range(len(x_stddev)):
        feature_vector.append(x_stddev[i])
    for i in range(len(x_skewness)):
        feature_vector.append(x_skewness[i])
    for i in range(len(x_kurtosis)):
        feature_vector.append(x_kurtosis[i])
    
    return feature_vector

#Extracts 64 BRIEF Features
def extractBriefFeatures(image):
        star = cv2.xfeatures2d.StarDetector_create()
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=16)
        vector_size = 10
        kps = star.detect(image)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        kps, des = brief.compute(image, kps)
        needed_size = (vector_size * 16)
        temparr=[]
        for i in range(len(kps)):
                for j in range(0,16):
                        temparr.append(des[i][j])
        while len(temparr) < needed_size:
                for i in range(0,16):
                        temparr.append(0)
        return temparr

input_file=csv.DictReader(open("/home/vanjani/Desktop/Melanoma/training_aug.csv"))

# loop over the training dataset
with open("/home/vanjani/Desktop/Melanoma/featurevector_brief.csv","w") as outfile:
        writer=csv.writer(outfile)
        
print ("[STATUS] Started extracting features")
count=0;
for row in input_file:
        count+=1;
        if count%50==0:
                print(count)

        file=train_path+"/"+row["image"]+".jpg"
        
        # read the training image in RGB
        image = cv2.imread(file)
        temparr = []
        
        # convert the image to various formats
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsvimage=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        labimage=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
        ycbimage=cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
        
        # extract haralick texture from the image in 4 orientations possible
        '''
        features = extract_features(gray)
        temparr=[]
        for i in range(len(features)):
            temparr.append(features[i])
        '''
        #extract color features (mean,standard deviation,skewness,kurtosis)
        '''
        mask=[]
        for i in range(len(image)):
            for j in range(len(image[i])):
                if image[i][j][0]<=10 and image[i][j][1]<=10 and image[i][j][2]<=2:
                    continue
                else:
                    mask.append((i,j))
                    
        #1. From RGB IMAGE
        features = extract_color(image,3,mask)
        for i in range(len(features)):
            temparr.append(features[i])

        #2. From HSV IMAGE
        features = extract_color(hsvimage,3,mask)
        for i in range(len(features)):
            temparr.append(features[i])

        #3. From Gray IMAGE
        features = extract_color(gray,1,mask)
        for i in range(len(features)):
            temparr.append(features[i])

        #4. From LAB IMAGE
        features = extract_color(labimage,3,mask)
        for i in range(len(features)):
            temparr.append(features[i])

        #5. From YCrCb IMAGE
        features = extract_color(ycbimage,3,mask)
        for i in range(len(features)):
            temparr.append(features[i])
        '''
        '''
        #extract LBP features
        hist = desc.describe(gray)
        for i in range(len(hist)):
            temparr.append(hist[i])
        '''
        
        #extract BRIEF features
        briefFeatures = extractBriefFeatures(image)
        for i in range(len(briefFeatures)):
            temparr.append(briefFeatures[i])

        with open("/home/vanjani/Desktop/Melanoma/featurevector_brief.csv","a") as outfile:
            writer=csv.writer(outfile)
            writer.writerow(temparr)
