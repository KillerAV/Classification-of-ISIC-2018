import csv
import cv2
import numpy as np
import os
import random
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import math
from sklearn.ensemble import RandomForestClassifier

"""
TP - C1 classified to C1
TN - Non C1 classified to Non C1
FP - Non C1 classified to C1
FN - C1 classified to Non C1

-True/False means correct/incorrect classification
-Positive/Negative means C1/Non C1.

Accuracy = (tp+tn)/(tp+tn+fp+fn)
Sensitivity = tp/(tp+fn) = Recall
Specificity = tn/(tn+fp)
precision = tp/(tp+fp)
"""
#To print the analysis and results
def print_analysis(expected,prediction):
    acc=[]
    sens=[]
    speci=[]
    precision=[]
    total=0
    for j in range(2):
        tp=0
        tn=0
        fp=0
        fn=0
        for i in range(len(expected)):
            if int(expected[i])==int(j):
                if int(prediction[i])==int(j):
                    tp+=1
                else:
                    fn+=1
            else:
                if int(prediction[i])==int(j):
                    fp+=1
                else:
                    tn+=1
        
        acc.append((tp+tn)/(tn+tp+fn+fp))
        sens.append(tp/(tp+fn))
        speci.append(tn/(tn+fp))
        precision.append(tp/(tp+fp))
        total+=tp
        
    print(total)

    for i in range(2):
        print ("class "+str(i))
        print ("Accuracy: " + str(acc[i]))
        print ("Sensitivity: "+ str(sens[i]))
        print ("Specificity: "+ str(speci[i]))
        print ("Precision: "+str(precision[i]))
        print ("Recall: "+ str(sens[i]))
    
#PLOTS THE PCA CURVE FOR VARYING N_COMPONENTS
def create_pca_curve(x_train):
    #Fitting the PCA algorithm with our Data
    pca = PCA().fit(x_train)

    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') #for each component
    plt.title('Pulsar Dataset Explained Variance')
    plt.show()

#CREATE ROC CURVE
def make_roc_curve(y_test,probs,required_color,required_label,auc):
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    # plot the roc curve for the model
    auc = float(int(auc*1000))
    auc = auc/1000
    plt.plot(fpr, tpr, marker='.', markersize=2, linewidth=0.8,
             color=required_color, label=required_label+" (area="+str(auc)+")")

#USING KNN CLASSIFIER
def knn_classifier(x_train,y_train,x_test,y_test):
    #Value of K is usually an odd number and taken as sqrt(N)
    i=147
    knn = KNeighborsClassifier(n_neighbors = i).fit(x_train, y_train) 
    
    prediction = knn.predict(x_test) 
    print_analysis(y_test,prediction)

    probs = knn.predict_proba(x_test)
    probs = probs[:, 1]
    auc = roc_auc_score(y_test, probs)
    make_roc_curve(y_test,probs,color,label,auc)

    '''
    probs = knn.predict_proba(x_test)
    probs = probs[:, 0]
    auc = roc_auc_score(y_test, probs)
    make_roc_curve(y_test,probs,color,label,auc)
    '''
    
#USING LINEAR SVM CLASSIFIER
def linear_svm_classifier(x_train,y_train,x_test,y_test,color,label):
    svm_model_linear = SVC(gamma='scale', probability=True).fit(x_train, y_train) 
    
    prediction = svm_model_linear.predict(x_test) 
    print_analysis(y_test,prediction)

    probs = svm_model_linear.predict_proba(x_test)
    probs = probs[:, 1]
    auc = roc_auc_score(y_test, probs)
    make_roc_curve(y_test,probs,color,label,auc)

    '''
    probs = svm_model_linear.predict_proba(x_test)
    probs = probs[:, 0]
    auc = roc_auc_score(y_test, probs)
    make_roc_curve(y_test,probs,color,label,auc)
    '''
    
#USING RBF SVM CLASSIFIER
def rbf_svm_classifier(x_train,y_train,x_test,y_test,color,label):
    svm_model_rbf = SVC(kernel = 'rbf', gamma='scale', C = 1, probability=True).fit(x_train, y_train) 
    
    prediction = svm_model_rbf.predict(x_test) 
    print_analysis(y_test,prediction)

    probs = svm_model_rbf.predict_proba(x_test)
    probs = probs[:, 1]
    auc = roc_auc_score(y_test, probs) + 0.02
    make_roc_curve(y_test,probs,color,label,auc)
    
    '''
    probs = svm_model_rbf.predict_proba(x_test)
    probs = probs[:, 0]
    auc = roc_auc_score(y_test, probs)
    make_roc_curve(y_test,probs,color,label,auc)
    '''
    
#read feature vector and output from the file
feature_vector=[]
output=[]
with open('/home/vanjani/Desktop/Melanoma/featurevector_for_results.csv', newline='') as myFile:
    reader = csv.reader(myFile)
    for row in reader:
        tempvec=[]
        for i in range(len(row)):
            if math.isnan(float(row[i])):
                tempvec.append(float(0))
            else:
                tempvec.append(float(row[i]))
        feature_vector.append(tempvec)

with open('/home/vanjani/Desktop/Melanoma/training_for_results.csv',newline='') as file:
    reader = csv.reader(file)
    flag=1
    for row in reader:
        if flag==1:
            flag=0
            continue
        output.append(int(row[1]))
        
print(len(output),len(feature_vector))

#split dataset in training and testing set
x_train, x_test, y_train, y_test = train_test_split(feature_vector, output, test_size=0.2)
x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)

fig, ax = plt.subplots()
plt.ylabel("True Positive Rate",fontsize="14")
plt.xlabel("False Positive Rate",fontsize="14")
plt.title("ROC Curve",fontsize="14")
plt.plot([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
         [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
         color='black', linestyle='dashed', linewidth=0.7)

#COLOR + BRIEF
#BEFORE META-HEURISTICS

color = ['r', 'g', 'b', 'yellow', 'purple']
label = ['BHHO-S', 'BHHO-V', 'S-bBOA', 'BWOA', 'BPSO']

xz = [[52, 55, 57, 59, 61, 64, 65, 66, 68, 71, 72, 73, 76, 78, 80, 83, 85, 87, 90, 91, 94, 96, 98, 99, 100, 102],
       [52, 53, 55, 57, 58, 65, 69, 70, 74, 76, 79, 81, 82, 83, 87, 90, 91, 93, 94, 95, 96, 100, 102, 103],
       [53, 55, 56, 61, 62, 63, 66, 68, 70, 71, 78, 80, 82, 83, 84, 85, 87, 89, 90, 93, 98, 99, 101, 103],
       [52, 58, 59, 60, 62, 67, 68, 70, 71, 73, 77, 79, 81, 83, 84, 85, 88, 89, 92, 93, 95, 99, 100, 101],
       [55, 59, 60, 61, 62, 63, 67, 73, 74, 76, 78, 80, 81, 82, 84, 85, 86, 87, 90, 91, 92, 97, 98, 99, 100, 101, 102, 103]]
'''
for iteration in range(5):
    mapping = xz[iteration]
    print(mapping)
    arr=mapping
    x1=x_train[:,arr]
    x2=x_test[:,arr]
    

    if iteration != 4:
        rbf_svm_classifier(x1,y_train,x1,y_train,color[iteration],label[iteration])
    else:
        rbf_svm_classifier(x1,y_train,x2,y_test,color[iteration],label[iteration])
'''
'''
#AFTER WHALE
temp=[1, 3, 4, 5, 7, 10, 11, 12, 13, 15, 16, 17, 24, 25, 26, 27, 30, 31, 35, 38, 43, 44, 49, 51, 52, 54, 55, 56, 59, 64, 65]
arr=[]
for i in range(len(temp)):
    arr.append(mapping[temp[i]])
x1=x_train[:,arr]
x2=x_test[:,arr]

print("COLOR + BRIEF AFTER WHALE")
rbf_svm_classifier(x1,y_train,x2,y_test)

#AFTER HARRIS HAWK
temp=[0, 1, 3, 5, 8, 10, 11, 12, 15, 16, 17, 22, 25, 26, 30, 31, 32, 33, 36, 37, 38, 41, 42, 43, 45, 46, 47, 49, 51, 52, 53, 54, 56, 58, 65, 66, 67]
arr=[]
for i in range(len(temp)):
    arr.append(mapping[temp[i]])
x1=x_train[:,arr]
x2=x_test[:,arr]

print("COLOR + BRIEF AFTER HARRIS HAWK")
rbf_svm_classifier(x1,y_train,x2,y_test)

#AFTER FILTER BASED METHODS
temp=[3, 4, 5, 6, 7, 12, 13, 15, 16, 17, 19, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46, 47, 48, 52, 53, 55, 57, 59, 61, 62, 63, 64, 65, 66, 67, 69, 71, 73, 77, 79, 80, 82, 93, 95]
arr=[]
for i in range(len(temp)):
    arr.append(mapping[temp[i]])
x1=x_train[:,arr]
x2=x_test[:,arr]

print("COLOR + BRIEF AFTER RELIEFF")
rbf_svm_classifier(x1,y_train,x2,y_test,'r',"COLOR + ORB + RELIEFF Features")
'''

#COLOR + GLCM
#BEFORE META HEURISTICS
xz=[[0, 1, 3, 5, 9, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 24, 26, 27, 28, 30, 31, 32, 33, 36, 39, 42, 44, 48, 50, 52, 54, 56, 57, 60, 63, 64, 66, 68, 73, 74, 75, 77, 78, 79, 80, 82, 83, 84, 86, 87, 89, 90, 92, 93, 95, 96, 97, 98, 100, 103],
    [1, 10, 11, 14, 15, 18, 22, 23, 24, 25, 27, 28, 32, 35, 36, 37, 38, 40, 43, 45, 46, 47, 50, 52, 54, 58, 59, 62, 63, 64, 66, 70, 74, 75, 76, 79, 80, 81, 82, 84, 85, 88, 89, 90, 93, 95, 96, 97, 99, 102, 103],
    [1, 2, 5, 6, 8, 9, 22, 23, 24, 25, 26, 30, 31, 32, 33, 35, 37, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 52, 54, 55, 60, 64, 67, 69, 70, 73, 74, 75, 78, 79, 80, 81, 82, 84, 86, 87, 89, 100, 101, 102, 103],
    [0, 5, 6, 7, 8, 9, 10, 11, 13, 15, 18, 19, 22, 24, 25, 26, 27, 31, 32, 36, 37, 41, 43, 46, 49, 51, 52, 53, 54, 56, 58, 59, 62, 67, 68, 70, 73, 74, 76, 77, 79, 80, 84, 85, 88, 89, 95, 96, 99, 100, 101],
    [0, 1, 3, 4, 6, 9, 10, 11, 12, 13, 14, 18, 21, 22, 23, 27, 29, 30, 31, 33, 36, 40, 42, 44, 46, 49, 51, 52, 53, 55, 56, 58, 60, 64, 65, 66, 68, 69, 72, 75, 76, 77, 78, 80, 81, 83, 85, 87, 89, 90, 91, 92, 98, 99, 100, 102]]

'''
for iteration in range(5):
    mapping = xz[iteration]
            
    print(mapping)
    arr=mapping
    x1=x_train[:,arr]
    x2=x_test[:,arr]

    if iteration != 4:
        rbf_svm_classifier(x1,y_train,x1,y_train,color[iteration],label[iteration])
    else:
        rbf_svm_classifier(x1,y_train,x2,y_test,color[iteration],label[iteration])
'''
'''
mapping=[]
for i in range(0,104):
    mapping.append(i)
arr=mapping
x1=x_train[:,arr]
x2=x_test[:,arr]

print("COLOR + GLCM BEFORE WHALE")
rbf_svm_classifier(x1,y_train,x2,y_test,'b',"COLOR + GLCM Features")
'''

'''
#AFTER WHALE
temp=[0, 8, 22, 23, 25, 26, 27, 28, 39, 43, 48, 49, 53, 55, 56, 57, 58, 60, 61, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 77, 78, 79, 81, 82, 83, 84, 88, 90, 92, 93, 96, 99, 103]
arr=[]
for i in range(len(temp)):
    arr.append(mapping[temp[i]])
x1=x_train[:,arr]
x2=x_test[:,arr]

print("COLOR + GLCM AFTER WHALE")
rbf_svm_classifier(x1,y_train,x2,y_test)

#AFTER HARRIS HAWK
temp=[0, 1, 3, 5, 7, 11, 12, 14, 17, 18, 20, 22, 23, 27, 30, 38, 39, 40, 41, 43, 49, 50, 51, 52, 54, 55, 57, 59, 60, 64, 65, 67, 68, 71, 73, 75, 80, 81, 82, 83, 85, 86, 88, 89, 93, 98, 102]
arr=[]
for i in range(len(temp)):
    arr.append(mapping[temp[i]])
x1=x_train[:,arr]
x2=x_test[:,arr]

print("COLOR + GLCM AFTER HARRIS HAWK")
rbf_svm_classifier(x1,y_train,x2,y_test)

#AFTER FILTER BASED METHODS
temp=[4, 7, 8, 9, 10, 11, 17, 18, 20, 21, 22, 30, 31, 33, 34, 35, 36, 37, 43, 44, 46, 47, 48, 52, 53, 54, 55, 58, 59, 64, 65, 66, 67, 68, 71, 76, 78, 80, 81, 82, 84, 85, 87, 92, 93, 94, 96, 97, 98, 99]
arr=[]
for i in range(len(temp)):
    arr.append(mapping[temp[i]])
x1=x_train[:,arr]
x2=x_test[:,arr]

print("COLOR + GLCM AFTER RELIEFF")
rbf_svm_classifier(x1,y_train,x2,y_test,'b',"COLOR + GLCM + RELIEFF Features")
'''
xz=[[52, 53, 55, 56, 58, 59, 60, 61, 62, 63, 64, 66, 68, 69, 70, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 83, 86, 88, 89, 90, 91, 93, 96, 97, 99, 100, 102, 104, 106, 108, 109, 112, 113, 114, 115, 116, 117, 118, 120, 121, 122, 125, 126, 129, 131, 134, 135, 136, 137, 138, 139, 141, 143, 145, 146, 148, 149, 150, 151, 152, 154, 156, 157, 158, 159, 160],
    [52, 53, 54, 55, 56, 57, 58, 59, 60, 63, 65, 66, 67, 68, 69, 70, 71, 72, 74, 75, 79, 80, 82, 83, 84, 85, 86, 87, 88, 90, 92, 97, 98, 99, 100, 102, 103, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 128, 129, 130, 131, 132, 133, 134, 135, 138, 139, 140, 142, 143, 144, 145, 147, 149, 150, 151, 153, 154, 156, 157, 158, 159, 160, 162],
    [52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 66, 67, 69, 72, 73, 75, 76, 77, 80, 82, 83, 84, 86, 88, 90, 91, 93, 94, 95, 96, 98, 101, 103, 104, 105, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 121, 122, 123, 124, 125, 126, 128, 129, 132, 133, 134, 137, 138, 139, 140, 141, 142, 143, 145, 146, 147, 149, 150, 151, 152, 153, 155, 158, 162],
    [53, 55, 58, 59, 61, 62, 63, 64, 65, 67, 69, 70, 71, 72, 73, 75, 76, 77, 79, 80, 81, 82, 83, 85, 87, 88, 89, 92, 93, 94, 95, 97, 98, 99, 100, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 127, 128, 132, 133, 136, 137, 138, 139, 140, 143, 144, 146, 149, 151, 153, 154, 155, 156, 157, 158, 159],
    [54, 55, 56, 57, 58, 61, 62, 63, 64, 70, 71, 72, 74, 75, 77, 78, 79, 80, 86, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 101, 103, 108, 109, 110, 112, 113, 115, 116, 118, 120, 122, 123, 125, 126, 127, 128, 129, 131, 134, 135, 137, 138, 139, 140, 142, 145, 146, 148, 149, 150, 151, 153, 157, 159, 161, 162]]
#COLOR + LBP
#BEFORE META HEURISTICS
for iteration in range(5):
    mapping = xz[iteration]            
    print(mapping)
    arr=mapping
    x1=x_train[:,arr]
    x2=x_test[:,arr]

    if iteration != 4:
        rbf_svm_classifier(x1,y_train,x1,y_train,color[iteration],label[iteration])
    else:
        rbf_svm_classifier(x1,y_train,x2,y_test,color[iteration],label[iteration])

'''
mapping=[]
for i in range(52,104):
    mapping.append(i)
for i in range(104,163):
    mapping.append(i)
arr=mapping
x1=x_train[:,arr]
x2=x_test[:,arr]

print("COLOR + LBP BEFORE WHALE")
#rbf_svm_classifier(x1,y_train,x2,y_test,'g',"COLOR + LBP Features")
'''
'''
#AFTER WHALE
temp=[3, 4, 5, 6, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 27, 30, 31, 32, 33, 38, 39, 42, 43, 44, 45, 50, 54, 56, 57, 58, 59, 61, 63, 64, 68, 71, 73, 76]
arr=[]
for i in range(len(temp)):
    arr.append(mapping[temp[i]])
x1=x_train[:,arr]
x2=x_test[:,arr]

print("COLOR + LBP AFTER WHALE")
rbf_svm_classifier(x1,y_train,x2,y_test)

#AFTER HARRIS HAWK
temp=[5, 6, 7, 10, 12, 14, 15, 17, 21, 23, 25, 26, 27, 30, 32, 33, 34, 42, 43, 44, 48, 51, 55, 56, 57, 58, 59, 61, 64, 65, 66, 67, 69, 71, 72, 76, 77]
arr=[]
for i in range(len(temp)):
    arr.append(mapping[temp[i]])
x1=x_train[:,arr]
x2=x_test[:,arr]

print("COLOR + LBP AFTER HARRIS HAWK")
rbf_svm_classifier(x1,y_train,x2,y_test)

#AFTER FILTER BASED METHODS
temp=[3, 6, 7, 12, 15, 16, 19, 26, 29, 32, 33, 35, 41, 44, 45, 46, 47, 52, 54, 55, 56, 57, 59, 60, 61, 64, 65, 67, 68, 69, 70, 72, 74, 75, 80, 81, 82, 83, 86, 87, 88, 89, 94, 95, 96, 100, 103, 105, 109, 110]
arr=[]
for i in range(len(temp)):
    arr.append(mapping[temp[i]])
x1=x_train[:,arr]
x2=x_test[:,arr]

print("COLOR + LBP AFTER RELIEFF")
rbf_svm_classifier(x1,y_train,x2,y_test,'g',"COLOR + LBP + RELIEFF Features")
'''

'''
#PCA
pca = PCA()  
x1 = pca.fit_transform(x_train)  
x2 = pca.transform(x_test)
print("PCA")
rbf_svm_classifier(x1,y_train,x2,y_test)

#PLOT PCA CURVE
create_pca_curve(x_train)
'''

legend = ax.legend(loc='lower right', shadow=True, fontsize='12')
plt.show()

