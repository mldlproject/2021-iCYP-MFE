# Import libraries
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor

#===========================================================================================
# Load data function
def load_data(task_name, batch_size):
    #--------------------
    train_data_x       = np.load("/source_cyp/data_processing/data_spliting/refined_data/splited_data/{}/train/data.npy".format(task_name))
    validation_data_x  = np.load("/source_cyp/data_processing/data_spliting/refined_data/splited_data/{}/val/data.npy".format(task_name))
    test_data_x        = np.load("/source_cyp/data_processing/data_spliting/refined_data/splited_data/{}/test/data.npy".format(task_name))
    #--------------------
    train_data_y       = np.load("/source_cyp/data_processing/data_spliting/refined_data/splited_data/{}/train/label.npy".format(task_name))
    validation_data_y  = np.load("/source_cyp/data_processing/data_spliting/refined_data/splited_data/{}/val/label.npy".format(task_name))
    test_data_y        = np.load("/source_cyp/data_processing/data_spliting/refined_data/splited_data/{}/test/label.npy".format(task_name))
    #--------------------
    train_dataset      = TensorDataset(Tensor(train_data_x).long(), Tensor(train_data_y))
    validation_dataset = TensorDataset(Tensor(validation_data_x).long(), Tensor(validation_data_y))
    test_dataset       = TensorDataset(Tensor(test_data_x).long(), Tensor(test_data_y))
    #--------------------
    print('Training data', train_data_x.shape)
    print('Validation data', validation_data_x.shape)
    print('Test data', test_data_x.shape)
    #--------------------
    training_loader    = DataLoader(train_dataset,      batch_size = batch_size, shuffle=True, worker_init_fn = np.random.seed(0)) # Fix random seed của torch.utils.data.DataLoader , worker_init_fn = np.random.seed(0)
    validation_loader  = DataLoader(validation_dataset, batch_size = batch_size, shuffle=False)
    test_loader        = DataLoader(test_dataset,       batch_size = batch_size, shuffle=False)
    #--------------------
    return training_loader, validation_loader, test_loader
    