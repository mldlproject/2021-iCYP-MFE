import csv, os
import random 
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from utils import save_result
################################################################################################
#  Splitting data for training dataset, validation dataset, test dataset
#-------------------------------------------------------------------------------
def np2vec(x, y, max_len, bit_size):
    data_x = []
    data_y = []
    for i in range(len(x)):
        fp = [0] * max_len
        n_ones = 0
        for j in range(bit_size):
            if x[i][j] == 1:
                fp[n_ones] = j+1
                n_ones += 1
        data_x.append(fp)
        data_y.append([y[i]])
        
    data_x = np.array(data_x, dtype=np.int32)
    data_y = np.array(data_y, dtype=np.int16).reshape(-1)
    return data_x, data_y

#-------------------------------------------------------------------------------
# Training set và Test set  (Test set is balanced)
def Train_test(dataX,dataY, dataR,seed):
    random.seed(seed)
    list_index0 = np.where(dataY == 0)
    list_index1 = np.where(dataY == 1)

    index0 = []
    while True:
        a = random.choice(list_index0[0].tolist())
        if a not in index0:
            index0.append(a)
        if len(index0) == 500:
            break
    index1 = []
    while True:
        a = random.choice(list_index1[0].tolist())
        if a not in index1:
            index1.append(a)
        if len(index1) == 500:
            break
    list_test = np.concatenate((np.array(index0), np.array(index1)), axis = 0).tolist()
    list_train = []
    for i in range(len(dataY)):
        if i not in list_test:
            list_train.append(i)
    X_train, X_test, Y_train, Y_test, R_train, R_test = [], [], [], [], [], []
    for index in list_test:
        R_test.append(dataR[index])
        X_test.append(dataX[index])
        Y_test.append(dataY[index])
    for index in list_train:
        R_train.append(dataR[index])
        X_train.append(dataX[index])
        Y_train.append(dataY[index])        
    return X_train, X_test, Y_train, Y_test, R_train, R_test

#-------------------------------------------------------------------------------
# Training set and Validation dataset
def Load_data(list_dataX, list_dataY, list_dataR, max_len, bit_size, task, seed):
    X_train, X_test, Y_train, Y_test, R_train, R_test = Train_test(list_dataX[task-1], list_dataY[task-1], list_dataR[task-1], seed) # test dataset
    X_train, X_val, Y_train, Y_val, R_train, R_val = train_test_split(X_train, Y_train, R_train, # training dataset, validation dataset
                                                stratify = Y_train, 
                                                test_size = 0.2, random_state = seed)
    X_train, X_val, X_test = np.array(X_train), np.array(X_val), np.array(X_test)
    Y_train, Y_val, Y_test = np.array(Y_train), np.array(Y_val), np.array(Y_test)
    R_train, R_val, R_test = np.array(R_train), np.array(R_val), np.array(R_test)
    
    
    train_data_x, train_data_y           = np2vec(X_train, Y_train, max_len, bit_size)
    train_data_r                         = R_train

    validation_data_x, validation_data_y = np2vec(X_val, Y_val, max_len, bit_size)
    validation_data_r                    = R_val

    test_data_x, test_data_y             = np2vec(X_test, Y_test, max_len, bit_size)
    test_R                               = R_test
    #-------------------------------------------------------------------------------
    train_dataset                        = TensorDataset(Tensor(train_data_x).long(), Tensor(train_data_r), Tensor(train_data_y))
    validation_dataset                   = TensorDataset(Tensor(validation_data_x).long(), Tensor(validation_data_r), Tensor(validation_data_y))
    test_dataset                         = TensorDataset(Tensor(test_data_x).long(), Tensor(test_R), Tensor(test_data_y))
    return train_dataset, validation_dataset, test_dataset