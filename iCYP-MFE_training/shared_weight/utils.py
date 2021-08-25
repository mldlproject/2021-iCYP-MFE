# Import libraries
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset

#===========================================================================================
# Loading data function
def load_data(batch_size):
    path_train = '/source_cyp/data_processing/data_shared_weight_extracting/refined_data/shared_weight_train'
    path_val   = '/source_cyp/data_processing/data_shared_weight_extracting/refined_data/shared_weight_val'
    #--------------------
    train_data_x = np.load(path_train + '/multi_refined_data.npy')
    train_data_y = np.load(path_train + '/multi_refined_labels.npy')
    #--------------------
    val_data_x  = np.load(path_val + '/multi_refined_data.npy')
    val_data_y  = np.load(path_val + '/multi_refined_labels.npy')
    #--------------------
    print('Training set: {}'.format(train_data_x.shape))
    print('Test set    : {}'.format(val_data_x.shape))
    train_dataset   = TensorDataset(Tensor(train_data_x).long(), Tensor(train_data_y[0]), Tensor(train_data_y[1]), Tensor(train_data_y[2]), Tensor(train_data_y[3]), Tensor(train_data_y[4]))
    val_dataset     = TensorDataset(Tensor(val_data_x).long(), Tensor(val_data_y[0]), Tensor(val_data_y[1]), Tensor(val_data_y[2]), Tensor(val_data_y[3]), Tensor(val_data_y[4]))
    #--------------------
    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    test_loader     = torch.utils.data.DataLoader(val_dataset,  batch_size = batch_size, shuffle=False)
    #--------------------
    return training_loader, test_loader

