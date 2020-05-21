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

input_file=csv.DictReader(open("/home/vanjani/Desktop/image/groundtruth.csv"))

train_path = "/home/vanjani/Desktop/image/datasetimages"
train_names = os.listdir(train_path)

with open("/home/vanjani/Desktop/image/training_aug.csv","w") as outfile:
                    writer=csv.writer(outfile)
count=0
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

def random_noise(image_array: ndarray):
    return skimage.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    return image_array[:, ::-1]

def brightness(image_array): 
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV) #convert to hsv
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * (np.random.uniform(1.1,1.3)) #scale channel V uniformly
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values
    rgb = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
    return rgb

def contrast(image_array):
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV) #convert to hsv
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * (np.random.uniform(0.7,0.9)) #scale channel V uniformly
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values
    rgb = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
    return rgb

def normalise(image_array):
    return cv2.normalize(image_array, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

def zoom(image, zoomSize):
    center = (int(image.shape[0]/2),int(image.shape[1]/2))
    cropScale = (int(center[0]/zoomSize), int(center[1]/zoomSize))
    zoom_image = image[(center[0]-cropScale[0]):(center[0] + cropScale[0]), (center[1]-cropScale[1]):(center[1] + cropScale[1])]
    zoom_image = cv2.resize(zoom_image,(image.shape[0],image.shape[1]))
    return zoom_image

def bilateral_filtering(image):
    blur = cv2.bilateralFilter(image,5,75,75)
    return blur

def upside_down_flip(image):
    return np.flipud(image)

def data_augmentation(image):
    augmented_images=[]
    random_noise_image=(random_noise(image)*255).astype('uint8')
    horizontal_flip_image=horizontal_flip(image)
    brightness_image=brightness(image)
    contrast_image=contrast(image)
    #normalise_image=(normalise(image)*255).astype('uint8')
    #zoom_image=zoom(image,1.5)
    #filter_image=bilateral_filtering(image)
    upside_down_image=upside_down_flip(image)
    augmented_images.append(random_noise_image)
    augmented_images.append(horizontal_flip_image)
    augmented_images.append(brightness_image)
    augmented_images.append(contrast_image)
    #augmented_images.append(normalise_image)
    #augmented_images.append(zoom_image)
    #augmented_images.append(filter_image)
    augmented_images.append(upside_down_image)
    return augmented_images

def k_means(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray_image = lab_image[:,:,0]
    
    resized_image=gray_image.reshape(gray_image.shape[0]*gray_image.shape[1], 1)/255
    kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=50, n_init=10, random_state=0).fit(resized_image)
    
    image_show = kmeans.cluster_centers_[kmeans.labels_]
    cluster_image = image_show.reshape(gray_image.shape[0], gray_image.shape[1], 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cluster_image = cv2.morphologyEx(cluster_image,cv2.MORPH_OPEN,kernel)

    min_pixel = np.min(cluster_image)
    
    masked_image = image
    for i in range(cluster_image.shape[0]):
        for j in range(cluster_image.shape[1]):
            if cluster_image[i][j] > min_pixel:
                masked_image[i][j] = (0,0,0)
    
    return masked_image

number_of_processed_images=0
count=0
aug_count=0
melanoma_count=0
non_melanoma_count=0
for row in input_file:
    file=train_path+"/"+row["image"]+".jpg"
    """
    PRE-PROCESSING
    """
    number_of_processed_images+=1
    if number_of_processed_images%100==0:
        print(melanoma_count,non_melanoma_count,number_of_processed_images)
        
    img=cv2.imread(file)
    image=cv2.resize(img,(256,256))
    
    if row["MEL"]=="1.0":
        output=0
        aug=data_augmentation(copy.deepcopy(image))
        path="./processeddataset_aug/melanoma/"
        for i in range(len(aug)):
            cv2.imwrite(os.path.join(path,"image_" + str(melanoma_count) + ".jpg"), k_means(aug[i]))
            melanoma_count+=1
        cv2.imwrite(os.path.join(path,"image_" + str(melanoma_count) + ".jpg"), k_means(image))
        melanoma_count+=1
        
    else:
        output=1
        path="./processeddataset_aug/non_melanoma/"
        cv2.imwrite(os.path.join(path,"image_" + str(non_melanoma_count) + ".jpg"), k_means(image))
        non_melanoma_count+=1
        
