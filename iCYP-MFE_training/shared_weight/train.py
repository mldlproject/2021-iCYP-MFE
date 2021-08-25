# Import libraries
import torch
import torch.nn as nn
from utils import load_data
from model import multi_task
from training_func import *

#===========================================================================================
# Set up GPU for running
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#===========================================================================================
# Set up training parameters
max_len        = 200 # (m)
bit_size       = 2048 # (n)
embedding_size = 200 #(k)
window_height  = 5
dropout_rate   = 0
num_filter     = 2048
batch_size     = 128
n_epoch        = 15
seed           = 1

#===========================================================================================
# Load dataset
training_loader, test_loader  = load_data(batch_size = batch_size)

# Create Model
model = multi_task(bit_size, embedding_size, max_len, window_height, num_filter, dropout_rate).to(device)

# Cross Entropy Loss 
criteron = nn.BCELoss()

# Adam Optimizer
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#==========================================================================================#                 
#                                    TRAINING MODEL                                        #           
#==========================================================================================#
train_loss_list, val_loss_list = [], []
loss_val_check = 10

print("Training Model")
for epoch in range(n_epoch):
    train_loss = train(epoch, model, criteron, optimizer, device, training_loader)
    val_loss   = validate(epoch, model, criteron, optimizer, device, test_loader)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    # Save model
    if val_loss < loss_val_check:
        # torch.save(model.state_dict(), './saved_pretrained_model/multi_model.pt')
        loss_val_check = val_loss

# save loss
# np.save('./loss/trainning_loss.npy', train_loss_list)
# np.save('./loss/validation_loss.npy', val_loss_list)

