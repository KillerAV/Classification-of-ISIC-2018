import csv
import cv2
import numpy as np
import os
import imutils
import skimage
import sys
from skimage import filters
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

train_path = "/home/vanjani/Desktop/image/datasetimages_temp"
train_names = os.listdir(train_path)

def k_means(image):
    cv2.imshow("(a)",image)
    cv2.waitKey(0)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    #cv2.imshow("(b)",lab_image)
    #cv2.waitKey(0)
    gray_image = lab_image[:,:,0]
    #cv2.imshow("(c)",gray_image)
    #cv2.waitKey(0)
    
    resized_image=gray_image.reshape(gray_image.shape[0]*gray_image.shape[1], 1)/255
    kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(resized_image)
    
    image_show = kmeans.cluster_centers_[kmeans.labels_]
    cluster_image = image_show.reshape(gray_image.shape[0], gray_image.shape[1], 1)

    kernel = np.ones((5,5), np.uint8)
    dic={}
    for i in range(256):
        for j in range(256):
            if cluster_image[i][j][0] in dic:
                dic[cluster_image[i][j][0]]+=1
            else:
                dic[cluster_image[i][j][0]]=1
    print(dic)
    min_pixel = np.min(cluster_image)

    cv2.imshow("S",cluster_image)
    cv2.waitKey(0)
    
    masked_image = image
    for i in range(cluster_image.shape[0]):
        for j in range(cluster_image.shape[1]):
            if cluster_image[i][j] > min_pixel:
                cluster_image[i][j] = 0
            else:
                cluster_image[i][j] = 255

    cv2.imshow("SC",cluster_image)
    cv2.waitKey(0)
                
    '''
    img_erosion = cv2.erode(cluster_image, kernel, iterations=1) 
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    cluster_image = img_dilation
    '''
    cv2.imshow("Original_without",cluster_image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    cluster_image = cv2.morphologyEx(cluster_image,cv2.MORPH_OPEN,kernel)

    for i in range(cluster_image.shape[0]):
        for j in range(cluster_image.shape[1]):
            if cluster_image[i][j]==0:
                masked_image[i][j] = (0,0,0)
    
    cv2.imshow("(d)",cluster_image)
    cv2.waitKey(0)
    cv2.imshow("(e)",masked_image)
    cv2.waitKey(0)
    
    return cluster_image
    

def thresholding(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_image=gray_image>125
    
    #Converting true false to integers.
    res=[]
    for i in range(len(threshold_image)):
        y=[]
        for j in range(len(threshold_image[i])):
            if threshold_image[i][j]:
                y.append(255)
            else:
                y.append(0)
        res.append(y)

    res=np.array(res)
    res=np.uint8(res)
    #cv2.imshow("(a)", gray_image)
    #cv2.waitKey(0)
    #cv2.imshow("(b)",res)
    #cv2.waitKey(0)
    return res

def edge_detection(image):
    kernel_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])
    gray_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_show = ndimage.convolve(gray_image, kernel_laplace, mode='reflect')
    #cv2.imshow("(a)", gray_image)
    #cv2.waitKey(0)
    #cv2.imshow("(b)", image_show)
    #cv2.waitKey(0)
    return image_show

for file in train_names:
    x=train_path+"/"+file
    print(x)
    image = cv2.imread(x)
    image = cv2.resize(image,(256,256))
    
    #THRESHOLDING
    #thresholding(image)
    
    #K-MEANS-CLUSTERING
    k_means(image)
    
    #EDGE DETECTION
    #edge_detection(image)

cv2.destroyAllWindows()
