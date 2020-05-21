import operator
import csv
import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.decomposition import PCA
import random
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

'''
feature_vector=[]
output=[]
with open('featurevector_aug.csv', newline='') as myFile:  
    reader = csv.reader(myFile)
    for row in reader:
        feature_vector.append(row)

with open('training_aug.csv',newline='') as myFile:
    reader = csv.reader(myFile)
    flag=0
    for row in reader:
        if flag==0:
            flag=1
            continue
        output.append(row[1])
'''

feature_vector=[]
class1=[]
class2=[]
output=[]
tempoutput=[]
with open('training_aug.csv',newline='') as myFile:
    reader = csv.reader(myFile)
    flag=0
    for row in reader:
        if flag==0:
            flag=1
            continue
        tempoutput.append(row[1])
        
with open('featurevector_aug.csv', newline='') as myFile:  
    reader = csv.reader(myFile)
    count=0
    for row in reader:
        if tempoutput[count]=="0":
            class1.append(row)
        elif tempoutput[count]=="1":
            class2.append(row)
        count+=1

print(len(class1),len(class2))


def random_data(arr,t):
    d={}
    for i in range(9000):
        x=random.randrange(0, len(arr))
        while x in d:
            x=random.randrange(0, len(arr))
        d[x]=1
        temp=[]
        for j in range(52,130):
            temp.append(arr[x][j])
        feature_vector.append(temp)
        output.append(t)

random_data(class1,0)
random_data(class2,1)
    
print(len(feature_vector),len(output))

#generating 30 random len(feature_vector[0]) size population containing ones/zeros
def initialize_pop():
    initial_population=np.zeros((30,len(feature_vector[0])))
    for i in range(initial_population.shape[0]):
        for j in range(initial_population.shape[1]):
            initial_population[i][j]=random.randint(0,1)
    return initial_population

# converting the values of output to int which were earlier in character
for i in range(len(output)):
    output[i]=int(output[i])

# converting the values of feature_vector to float which were earlier in character
for i in range(len(feature_vector)):
    for j in range(len(feature_vector[i])):
        feature_vector[i][j]=float(feature_vector[i][j])

# split the dataset in the ratio 70 and 30
x_train, x_test, y_train, y_test = train_test_split(feature_vector, output, test_size=0.3)

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

def get_losses(population,flag):  # The rank function based on fitness value
    losses=[]
    global x_train,y_train
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    mean_of_individuals=np.mean(population,axis=1)
    count=0
    for individual in population:
        y=[]
        arr=[]
        if(flag==1):
            for i in range(len(individual)):
                if individual[i]==1.:
                    arr.append(i)
            
            if len(arr)==0:
                losses.append(int(10000000000000000000000000000))
                continue
        else:
            for i in range(len(individual)):
                if individual[i]>=mean_of_individuals[count]:
                    arr.append(i)
            
            if len(arr)==0:
                losses.append(int(10000000000000000000000000000))
                continue

        x=x_train[:,arr]
        y=y_train
        clf=svm.SVC(gamma='scale')
        clf.fit(x,y)
        predicted=clf.predict(x)

        #select the features based on this individual to get xtrainnew 
        #loss calculated should be mean sq error
        #error = len(y) - np.sum(np.array(y) == np.array(predicted))
        #error = cross_entropy(predicted, y)
        error = mean_squared_error(y, predicted) 
        losses.append(error)
        count=count+1
        
    zip1 = zip(losses, population)
    sorted_results = sorted(zip1, key=operator.itemgetter(0))
    sorted_pop = [x for _, x in sorted_results]
    sorted_losses = [_ for _, x in sorted_results]
    return sorted_pop, sorted_losses

#whale algorithm
def Whale_Optimization():
    
    whale_pop=initialize_pop()
    population=whale_pop #generation of initial population
    max_iterations=30
    l=np.random.uniform(low=-1,high=1)
    sorted_pop,sorted_losses=get_losses(population,1) #population will already be in 1/0 form
    population=sorted_pop
    x_star=sorted_pop[0] #population with best fitness
    best_fitness=sorted_losses[0]
    #best fitness is one with  least error

    p=np.random.uniform(low=0,high=1)
    b=1 #acc to source code
    r1=np.random.uniform(low=0,high=1)
    r2=np.random.uniform(low=0,high=1)
    a=2
    A=2*a*r1-a
    C=2*r2
    D=population[0]
    
    for iterations in range(max_iterations):
        
        a=2-((2)*((iterations+1)/max_iterations))
        print(iterations)
        for i in range(30):
            l=np.random.uniform(low=-1,high=1)
            p=np.random.uniform(low=0,high=1)
            r1=np.random.uniform(low=0,high=1)
            r2=np.random.uniform(low=0,high=1)
            A=2*a*r1-a
            C=2*r2
            p=np.random.uniform(low=0,high=1)
            Absolute_value_A=abs(A)
            if(p<0.5):
                if(Absolute_value_A>=1):
                    rand_int_exceptI=np.random.randint(low=0, high=30)
                    while i==rand_int_exceptI :
                        rand_int_exceptI=np.random.randint(low=0, high=30)
                    D=abs(C*population[rand_int_exceptI]-population[i])
                    population[i]=population[rand_int_exceptI]-A*D
                else:
                    D=abs(C*(x_star)-population[i])
                    population[i]=x_star-A*D   
            else:
                D=abs(x_star-population[i])
                population[i]=np.exp(b*l)*math.cos(2*math.pi*l)*D+x_star
      
        
        sorted_pop, sorted_losses = get_losses(population,0)
        population = sorted_pop
        if best_fitness > sorted_losses[0]:
            x_star = population[0]
            best_fitness = sorted_losses[0]
        
        print("FITNESS: " + str(best_fitness))
        
        arr=[]
        final_mean=np.mean(np.array(x_star))
        for index in range(len(x_star)):
            if(x_star[index]>=final_mean):
                arr.append(index)

        print(arr)
        print()
        
    final_mean=np.mean(np.array(x_star))
    final_ans=[]
    ansPrint=[]
    #although this doent seem to be needed but just used it anyway 
    for index in range(len(x_star)):
        if(x_star[index]>=final_mean):
            ansPrint.append(index)
            final_ans.append(1)
        else:
            final_ans.append(0)    

    print(ansPrint)
    return final_ans,sorted_losses[0]

def getAccuracy(population):
    #get new x_train and new x_test in accordance to the obtained population using whale metaheuristic
    
    # Training accuracy:
    # obtain y_pred_training using the new obtained x_train
    global x_train,x_test,y_train,y_test
    x_train=np.array(x_train)
    x_test=np.array(x_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    x=[]
    y=[]
    arr=[]
    individual=population
    for i in range(len(individual)):
            if individual[i]==1:
                arr.append(i)
    x=x_train[:,arr]
    y=y_train
    clf=svm.SVC(gamma='scale')
    clf.fit(x,y)

    # Test accuracy:
    x1=x_test[:,arr]
    y1=y_test

    predicted=clf.predict(x1)
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y1, predicted)
    accuracy_test = (cm[0][0] + cm[1][1]) / np.sum(cm)
    print("TEST ACCURACY = ", accuracy_test,"\n")
    return accuracy_test


# main starts here,calling whale
best_test_acc=0
best_pop=[]
best_arr=[]
for j in range(5):
    print ("ITTERATION "+str(j))
    new_pop,new_loss=Whale_Optimization()
    curr_test_acc=getAccuracy(new_pop)
    print("BEST INDIVIDUAL IN WHALE: ")
    arr=[]
    for i in range(len(new_pop)):
            if new_pop[i]==1:
                arr.append(i)
    print(arr)
    print(curr_test_acc)
    if(curr_test_acc>best_test_acc):
        best_test_acc=curr_test_acc
        best_pop=new_pop
        best_arr=arr
# "FINALLY WE WILL HAVE OUR REQUIRED POPULATION IN BEST POPULATION"
print(best_arr)

