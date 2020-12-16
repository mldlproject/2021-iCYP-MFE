# Import Python libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

from model import Multi_task
from split_data import torch_data_loading

# Set up GPU for running
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

##########################################################################################
# Set up training parameters
max_len        = 200 # (m)
bit_size       = 2048 # (n)
embedding_size = 200 #(k)
window_height  = 5
dropout_rate   = 0
num_filter     = 4096
batch_size     = 128
seed           = 5
##########################################################################################

# Load dataset

print("Loadding dataset")
dataX = np.load('/content/drive/My Drive/Predict_task/Multi_task/data/dataX.npy')
dataY = np.load('/content/drive/My Drive/Predict_task/Multi_task/data/dataY.npy')
train_dataset, test_dataset = torch_data_loading(dataX, dataY, max_len, bit_size, seed)

training_loader     = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader         = torch.utils.data.DataLoader(test_dataset,  batch_size = batch_size, shuffle = False)
##########################################################################################

# Create Model
model = Multi_task(bit_size, embedding_size, max_len, window_height, num_filter, dropout_rate).to(device)
# Cross Entropy Loss 
criteron = nn.BCELoss()
# Adam Optimizer
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

##########################################################################################                 
###                  DEFINE TRAINING, VALIDATION, AND TEST FUNCTION                    ###           
##########################################################################################

# Training Function
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(training_loader):
        dataX = data[0].to(device)
        Z, task1, task2, task3, task4, task5 = model(dataX)

        list_labels = [data[i].to(device) for i in range(1,6)]
        list_task = [task1.view_as(list_labels[0]), task2.view_as(list_labels[1]), task3.view_as(list_labels[2]), task4.view_as(list_labels[3]), task5.view_as(list_labels[4])]

        #------------------- 
        optimizer.zero_grad()
        #------------------- 
        loss = 0
        for i in range(len(list_task)):
            loss += criteron(list_task[i], list_labels[i]) 
        loss = loss/len(list_task) 
        loss.backward()
        train_loss += loss.item()*len(dataX) #(loss.item is the average loss of training batch)
        optimizer.step() 
    
    print('====> Epoch: {} Average Train Loss: {:.4f}'.format(epoch, train_loss / len(training_loader.dataset)))
    train_loss = (train_loss / len(training_loader.dataset) )

    return train_loss

##########################################################################################
# Test Function
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for _, data in enumerate(test_loader): 
            dataX = data[0].to(device)
            Z, task1, task2, task3, task4, task5 = model(dataX)

            list_labels = [data[i].to(device) for i in range(1,6)]
            list_task = [task1.view_as(list_labels[0]), task2.view_as(list_labels[1]), task3.view_as(list_labels[2]), task4.view_as(list_labels[3]), task5.view_as(list_labels[4])]
            loss = 0
            for i in range(len(list_task)):
                loss += criteron(list_task[i], list_labels[i])
            loss = loss/len(list_task)
            test_loss += loss.item()*len(dataX)

    test_loss  = (test_loss / len(test_loader.dataset) )
    print('====> Average Test Loss: {:.4f}'.format(test_loss))

    return test_loss

##########################################################################################                 
###                                  TRAINING MODEL                                    ###           
##########################################################################################

test_loss_list = []
train_loss_list = []
count = 10

print("Training Model")
for epoch in range(1, 20):
    train_loss = train(epoch)
    test_loss = test(epoch)

    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    # Save model
    if test_loss < count:
        # torch.save(model, '/content/drive/My Drive/Predict_task/Multi_task/pretrain_model/model_seed' + str(seed) + '/Multi_model.pt')
        count = test_loss


plt.plot(train_loss_list, label='Training Loss')  
plt.plot(test_loss_list, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-entropy Loss avenge all task')    
plt.legend()  
plt.savefig('/content/drive/My Drive/Predict_task/Multi_task/plot.png')
plt.show()
