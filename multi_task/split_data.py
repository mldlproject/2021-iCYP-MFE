# Xử lý dữ liệu 
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import random

#----------------------------------------------------
def np2vec(x, y, max_len, bit_size):
    data_x = []
    data_y0, data_y1, data_y2, data_y3, data_y4  = [], [], [], [], []
    data_y = [data_y0, data_y1, data_y2, data_y3, data_y4]
    for i in range(len(x)):
        fp = [0] * max_len
        n_ones = 0
        for j in range(bit_size):
            if x[i][j] == 1:
                fp[n_ones] = j+1
                n_ones += 1
        data_x.append(fp)
        # data_y.append([y[i]])
    for i in range(len(y)):
        for j in range(len(x)):
            data_y[i].append([y[i][j]])
        
    data_x = np.array(data_x, dtype=np.int32)
    data_y0 = np.array(data_y0, dtype=np.int16).reshape(-1)
    data_y1 = np.array(data_y1, dtype=np.int16).reshape(-1)
    data_y2 = np.array(data_y2, dtype=np.int16).reshape(-1)
    data_y3 = np.array(data_y3, dtype=np.int16).reshape(-1)
    data_y4 = np.array(data_y4, dtype=np.int16).reshape(-1)
    data_y = [data_y0, data_y1, data_y2, data_y3, data_y4] 
    return data_x, data_y

#--------------------------------------------------------------------
def Split_train_test(dataX,list_dataY, seed):
    random.seed(seed) # RANDOM SEED 
    list_index = np.arange(dataX.shape[0])
    # List index của tập test
    list_test = []

    # Chia 500 mẫu cho tập validation
    while True:
        a = random.choice(list_index.tolist())
        if a not in list_test:
            list_test.append(a)
        if len(list_test) == 500:
            break

    # List index của tập train 
    list_train = []
    for i in range(len(dataX)):
        if i not in list_test:
            list_train.append(i)
    
    # Split training dataset and validation dataset
    X_train, X_test, Y_train1, Y_test1, Y_train2, Y_test2, Y_train3, Y_test3, Y_train4, Y_test4, Y_train5, Y_test5 = [], [], [], [], [], [], [], [], [], [], [], []
    Y_train = [Y_train1, Y_train2, Y_train3, Y_train4, Y_train5]
    Y_test = [Y_test1, Y_test2, Y_test3, Y_test4, Y_test5]

    for index in list_train:
        X_train.append(dataX[index]) # vector index for validation datase
    for index in list_test:
        X_test.append(dataX[index]) # vector index for validation dataset

    for i in range(len(list_dataY)):
        for index in list_test:
            Y_test[i].append(list_dataY[i][index]) # Label tương ứng với mỗi task(validation dataset)        
        for index in list_train:
            Y_train[i].append(list_dataY[i][index]) # Label tương ứng với mỗi task(training dataset)   

    return X_train, X_test, Y_train, Y_test

def torch_data_loading(dataX, list_dataY, max_len = 220, bit_size = 2048, seed= 1):
    # Load data

    X_train, X_test, Y_train, Y_test = Split_train_test(dataX, list_dataY, seed)
    #-------------------------------------------------------------------------------
    
    train_data_x, train_data_y           = np2vec(X_train, Y_train, max_len, bit_size)
    test_data_x, test_data_y             = np2vec(X_test, Y_test, max_len, bit_size)
    #-------------------------------------------------------------------------------

    train_dataset                        = TensorDataset(Tensor(train_data_x).long(), Tensor(train_data_y[0]), Tensor(train_data_y[1]), Tensor(train_data_y[2]), Tensor(train_data_y[3]), Tensor(train_data_y[4]))
    test_dataset                         = TensorDataset(Tensor(test_data_x).long(), Tensor(test_data_y[0]), Tensor(test_data_y[1]), Tensor(test_data_y[2]), Tensor(test_data_y[3]), Tensor(test_data_y[4]))
    
    return train_dataset, test_dataset
