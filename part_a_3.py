import time as t
import numpy as np
import pandas as pd
from scipy import sparse
import string
import sys
import os

rate = float(sys.argv[2])
iterations = int(sys.argv[3])
batch_size = int(sys.argv[4])

train_read = sys.argv[5]
train_path = os.path.abspath(train_read)
path_train = os.path.dirname(train_path)
os.chdir(path_train)

raw_read = sys.argv[6]
raw_path = os.path.abspath(raw_read)
path_raw = os.path.dirname(raw_path)
os.chdir(path_raw)

test_read = sys.argv[7]
test_path = os.path.abspath(test_read)
path_test = os.path.dirname(test_path)
os.chdir(path_test)

raw = ((pd.read_csv(raw_read, header = None)).values)[:,0]
x_train = (pd.read_csv(train_read, header = None, na_filter = False, low_memory = False)).values
x_test = (pd.read_csv(test_read, header = None, na_filter = False, low_memory = False)).values
y_train = sparse.csr_matrix((x_train[:,0]), dtype = int).T
y_test = sparse.csr_matrix(x_test[:,0], dtype = int).T

for i in range(x_train.shape[0]):
    x_train[i,1] = (''.join([j.lower() for j in x_train[i,1] if j in string.ascii_letters + '\'- '])).split(" ")

for i in range(x_test.shape[0]):
    x_test[i,1] = (''.join([j.lower() for j in x_test[i,1] if j in string.ascii_letters + '\'- '])).split(" ")

calc = {}
for i in range(len(raw)):
    calc.update({raw[i]: i})

train_data = sparse.lil_matrix((x_train.shape[0], len(raw)), dtype = int)
for i in range(x_train.shape[0]):
    for j in range(len(x_train[i,1])):
        if x_train[i,1][j] in calc.keys():
            train_data[i,calc[x_train[i,1][j]]] += 1
train_data = train_data.tocsr()
rate = 10
test_data = sparse.lil_matrix((x_test.shape[0], len(raw)), dtype = int)
for i in range(x_test.shape[0]):
    for j in range(len(x_test[i,1])):
        if x_test[i,1][j] in calc.keys():
            test_data[i,calc[x_test[i,1][j]]] += 1
test_data = test_data.tocsr()

def g(x):
    return sparse.csr_matrix(1.0/(1 + np.exp(-x.todense())))

def rounder(x):
    for i in range(x.shape[0]):
        if(x[i,0] > 0.5):
            x[i,0] = 1
        else:
            x[i,0] = 0
    return(x)

def error(test_data, y_test, w0):
    return np.sum(abs(rounder(g(test_data*w0))-y_test))
#    return np.sum(abs(g(test_data*w0))-y_test)

def gradient(train_data, y_train, w0, lamda):
    temp1 = (train_data.T)*(g(train_data*w0)-y_train)
    temp2 = lamda*w0
    return ((temp1+temp2)/train_data.shape[0])   

def logistic(train_data, y_train, lamda, rate):
    w0 = sparse.csr_matrix([0]*len(raw)).T
    w1 = sparse.csr_matrix.copy(w0)
    for i in range(3*iterations):
        w1 = w0 - (rate/((i+1)**(0.5)))*gradient(train_data, y_train, w0, lamda)
        w0 = sparse.csr_matrix.copy(w1)
    return(w0)

def divider(train_data, y_train):
    x_cross = []
    y_cross = []
    step = round((train_data.shape[0])/10)
    for i in range(0,train_data.shape[0],step):
        x_cross.append(train_data[i:i+step,:])
    for i in range(0,y_train.shape[0],step):
        y_cross.append(y_train[i:i+step,:])
    return x_cross, y_cross

def cross_acc(train_data, y_train, lamda, rate):
    accuracy = []
    x, y = divider(train_data, y_train)
    for i in range(10):
        print(i)
        train = list(x)
        test = list(y)
        del train[i]
        del test[i]
        for j in range(0,8):
            train[j+1] = sparse.vstack([train[j], train[j+1]])
            test[j+1] = sparse.vstack([test[j], test[j+1]])
        train_x = sparse.csr_matrix(train[8])
        test_y = sparse.csr_matrix(test[8])
        w0 = logistic(train_x, test_y, lamda, rate)
        accuracy.append(error(x[i], y[i], w0))
    acc = 0
    for i in range(len(accuracy)):
        acc += accuracy[i]
    return (acc/len(accuracy))


#lamda = [10**(-16), 10**(-15), 10**(-14), 10**(-13), 10**(-12), 10**(-11), 10**(-10), 10**(-9), 10**(-8), 10**(-7)]
#for i in range(len(lamda)):
#    print(i)
#    print(str(lamda[i]) + '---------' + str(cross_acc(train_data,  y_train, lamda[i], 10)))

lamda = 10**(-14)

w0 = logistic(train_data, y_train, lamda, rate)

output = np.array(rounder(g(test_data*w0)).todense())

out_write = sys.argv[8]
path_out = os.path.abspath(out_write)
out_path = os.path.dirname(path_out)
os.chdir(out_path)
np.savetxt(out_write, output)