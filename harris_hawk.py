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

#Data-set Preparation
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
    for i in range(7000):
        x=random.randrange(0, len(arr))
        while x in d:
            x=random.randrange(0, len(arr))
        d[x]=1
        temp=[]
        '''
        #FOR BRIEF
        for j in range(52,104):
            temp.append(arr[x][j])
        for j in range(130,146):
            temp.append(arr[x][j])
        '''
        '''
        #FOR KAZE
        for j in range(52,104):
            temp.append(arr[x][j])
        for j in range(146,210):
            temp.append(arr[x][j]) 
        ''' 
        #FOR GLCM
        for j in range(0,104):
            temp.append(arr[x][j])
        '''
        #FOR LBP
        for j in range(52,104):
            temp.append(arr[x][j])
        for j in range(104,130):
            temp.append(arr[x][j])
        '''
        feature_vector.append(temp)
        output.append(t)

random_data(class1,0)
random_data(class2,1)
    
print(len(feature_vector),len(output))

population_size = 30
#generating population_size random len(feature_vector[0]) size population containing ones/zeros
def initialize_pop():
    initial_population=np.zeros((population_size,len(feature_vector[0])))
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
x_train=np.array(x_train)
y_train=np.array(y_train)
def get_losses(population):  # The rank function based on fitness value
    losses=[]
    global x_train,y_train
    for individual in population:
        y=[]
        arr=[]
        for i in range(len(individual)):
            if individual[i]==1:
                arr.append(i)
            
        if len(arr)==0:
            losses.append(int(10000000000000000000000000000))
        else:
            x=x_train[:,arr]
            y=y_train
            clf=svm.SVC(gamma='scale')
            clf.fit(x,y)
            predicted=clf.predict(x)
            error = mean_squared_error(y, predicted) 
            losses.append(error)
        
    zip1 = zip(losses, population)
    sorted_results = sorted(zip1, key=operator.itemgetter(0))
    sorted_pop = [x for _, x in sorted_results]
    sorted_losses = [_ for _, x in sorted_results]
    return sorted_pop, sorted_losses

def LF(x):
    u = np.random.normal(0,1,x)     #mean=0 , SD=1
    v = np.random.normal(0,1,x)
    beta=1.5
    sigma = pow((math.gamma(1+beta) * math.sin(math.pi * beta / 2)) / (math.gamma((1+beta)/2) * beta * pow(2, (beta-1)/2)), 1/beta)
    return u * 0.01 * sigma / pow(abs(v), 1/beta)

def get_fitness(individual):
    global x_train,y_train
    arr=[]
    for i in range(len(individual)):
        if individual[i]==1:
            arr.append(i)
            
    if len(arr)==0:
        return int(10000000000000000000000000000)
    else:
        x=x_train[:,arr]
        y=y_train
        clf=svm.SVC(gamma='scale')
        clf.fit(x,y)
        predicted=clf.predict(x)
        error = mean_squared_error(y, predicted)
        return error
    
def decide_population(Y, Z, X):
    fitness_X = get_fitness(X)
    fitness_Y = get_fitness(Y)
    fitness_Z = get_fitness(Z)
    if fitness_Y < fitness_X:
        return Y
    elif fitness_Z < fitness_X:
        return Z
    else:
        return X

def transfer_function_sigmoidal(population, iteration, max_iterations):
    torque_max=4
    torque_min=0.01
    torque = (1-iteration/max_iterations) * torque_max + iteration * torque_min/max_iterations
    for i in range(len(population)):
        for j in range(len(population[i])):
            r = np.random.uniform(0,1)
            F_x = 1 / (1 + np.exp(-population[i][j]/torque))
            if F_x > r:
                population[i][j] = 1
            else:
                population[i][j] = 0
    return population
    
#HHO Algorithm
def Harris_Hawk_Optimization():
    hawk_pop=initialize_pop()
    population=hawk_pop #generation of initial population
    max_iterations=50
    best_fitness=1000000000000000000
    X_rabbit=np.array([])
    for iteration in range(max_iterations):
        print("ITERATION:", iteration)
        print()
        sorted_pop,sorted_losses = get_losses(population)
        population=sorted_pop
        if(best_fitness>sorted_losses[0]):
            X_rabbit=sorted_pop[0]          #population with best fitness
            best_fitness=sorted_losses[0]   #best fitness is one with  least error

        print("FITNESS: " + str(best_fitness))
        arr=[]
        for index in range(len(X_rabbit)):
            if X_rabbit[index]==1:
                arr.append(index)
        print(arr)
        print()
        
        newpopulation = population
        for i in range(population_size):
            E0 = np.random.uniform(low=-1,high=1)
            J = 2*(1-np.random.uniform(low=0,high=1))
            E = 2*E0*(1-iteration/max_iterations)
            X_mean = np.mean(population,axis=0)
            
            #EXPLORATION PHASE
            if abs(E)>=1:
                q=np.random.uniform(0,1)
                if q>=0.5:
                    random_hawk=np.random.randint(low=0, high=population_size)
                    while i==random_hawk:
                        random_hawk=np.random.randint(low=0, high=population_size)
                    r1=np.random.uniform(0,1)
                    r2=np.random.uniform(0,1)
                    newpopulation[i] = population[random_hawk] - r1 * abs(population[random_hawk] - 2 * r2 * population[i])
                else:
                    r3=np.random.uniform(0,1)
                    r4=np.random.uniform(0,1)
                    LB=0
                    UB=1
                    newpopulation[i] = (X_rabbit - X_mean) - r3 * (LB + r4 * (UB - LB))
            #EXPLOITATION PHASE
            else:
                r=np.random.uniform(0,1)
                if r>=0.5 and abs(E)>=0.5:
                    delta_X = X_rabbit - population[i]
                    newpopulation[i] = delta_X - E * abs(J * X_rabbit - population[i])
                elif r>=0.5 and abs(E)<0.5:
                    delta_X = X_rabbit - population[i]
                    newpopulation[i] = X_rabbit - E * abs(delta_X)
                elif r<0.5 and abs(E)>=0.5:
                    Y = X_rabbit - E * abs(J * X_rabbit - population[i])   
                    S = np.random.random(len(X_rabbit))
                    Z = Y + LF(len(X_rabbit)) * S
                    newpopulation[i] = decide_population(Y, Z, population[i])
                else:
                    Y = X_rabbit - E * abs(J * X_rabbit - X_mean)   
                    S = np.random.random(len(X_rabbit))
                    Z = Y + LF(len(X_rabbit)) * S
                    newpopulation[i] = decide_population(Y, Z, population[i])

        population = transfer_function_sigmoidal(newpopulation,iteration,max_iterations)
    
    final_ans=[]
    for index in range(len(X_rabbit)):
        if(X_rabbit[index]==1):
            final_ans.append(index)    

    print(final_ans)
    return final_ans,sorted_losses[0]

def getAccuracy(individual):
    global x_train,x_test,y_train,y_test
    x_train=np.array(x_train)
    x_test=np.array(x_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    x=[]
    y=[]

    x=x_train[:,individual]
    y=y_train
    clf=svm.SVC(gamma='scale')
    clf.fit(x,y)

    # Test accuracy:
    x1=x_test[:,individual]
    y1=y_test

    predicted=clf.predict(x1)
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y1, predicted)
    accuracy_test = (cm[0][0] + cm[1][1]) / np.sum(cm)
    print("TEST ACCURACY = ", accuracy_test,"\n")
    return accuracy_test

# main starts here, calling HHO 
best_test_acc=0
best_arr=[]
for j in range(1):
    print ("ITERATION "+str(j))
    new_individual, new_loss = Harris_Hawk_Optimization()
    curr_test_acc = getAccuracy(new_individual)
    
    print("BEST INDIVIDUAL IN HARRIS HAWK: ")
    if(curr_test_acc>best_test_acc):
        best_test_acc=curr_test_acc
        best_arr=new_individual
    print(best_arr)
    print("BEST FITNESS ACCURACY ON TEST DATA:")
    print(best_test_acc)
    
    
# "FINALLY WE WILL HAVE OUR REQUIRED POPULATION IN BEST POPULATION"
print(best_arr)

