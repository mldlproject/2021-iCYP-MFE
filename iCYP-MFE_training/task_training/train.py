# Import libraries
import torch
import torch.nn as nn
import numpy as np
from model import single_task
from training_func import train, validate, test
from utils import *
from dataset import *
from pre_trainlayer import multi_task

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
batch_size     = 64
task_name      = 'cyp1a2' # 'cyp2c9', 'cyp2c19', 'cyp2d6', 'cyp3a4'
n_epoch        = 10
use_pretrain_layer = True # True: training multi-task, False: training single task

#===========================================================================================
# Loading data 
training_loader, validation_loader, test_loader = load_data(task_name, batch_size)

#===========================================================================================
# Load pretrain model(model multi_task) 
model_multi = multi_task(bit_size, embedding_size, max_len, window_height, num_filter, dropout_rate).to(device)
model_multi.load_state_dict(torch.load('./saved_pretrained_model/multi_model.pt'))
model_multi.eval()

#===========================================================================================
# Create Model (Single model)
model = single_task(model_multi, bit_size, embedding_size, max_len, window_height, num_filter, dropout_rate, share_layer= use_pretrain_layer).to(device)

# Adam optimizer
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Cross Entropy Loss 
criteron = nn.BCELoss()

#==========================================================================================#                 
#                                    TRAINING MODEL                                        #           
#==========================================================================================#
training_loss_list, validation_loss_list, test_loss_list = [], [], []
test_prob_pred, val_prob_pred = [], []

print("Training model for task : {}".format(task_name))
val_loss_check = 10
for epoch in range(n_epoch):
    train_results = train(epoch, model, criteron, optimizer, device, training_loader)
    validation_results = validate(epoch, model, criteron, device, validation_loader)
    if validation_results[0] < val_loss_check:
        val_loss_check = validation_results[0]
        # torch.save(model.state_dict(), "path for save model")
    test_results = test(epoch, model, criteron, device, test_loader)
    #------------------- 
    training_loss_list.append(train_results)
    #-------------------
    validation_loss_list.append(validation_results[0])
    val_prob_pred.append(validation_results[1])
    #-------------------
    test_loss_list.append(test_results[0])
    test_prob_pred.append(test_results[1])

#==========================================================================================#                 
#                                    EVALUATING MODEL                                      #           
#==========================================================================================#
# perform test_dataset
probs  = get_prob(test_prob_pred, np.argmin(validation_loss_list))
labels = np.load('/source_cyp/data_processing/data_spliting/refined_data/splited_data/{}/test/label.npy'.format(task_name))
printPerformance(labels, probs)
