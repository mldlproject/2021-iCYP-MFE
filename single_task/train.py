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

from model import Single_task
from utils import *
from Split_data import Load_data
from pre_trainlayer import Multi_task

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

##########################################################################################
# Set up training parameters
max_len        = 200 # (m)
bit_size       = 2048 # (n)
embedding_size = 200 #(k)
window_height  = 5
dropout_rate   = 0
num_filter     = 4096
batch_size     = 128
seed           = 1
task           = 1
# Loading dataset: task 1 và seed = 1
print("Loadding data")
list_dataX = np.load('/content/drive/MyDrive/Predict_task/Single_task/data/dataX.npy', allow_pickle=True) 
list_dataY = np.load('/content/drive/MyDrive/Predict_task/Single_task/data/dataY.npy', allow_pickle=True)
list_dataR = np.load('/content/drive/MyDrive/Predict_task/Single_task/data/dataR.npy', allow_pickle=True)
train_dataset, validation_dataset, test_dataset = Load_data(list_dataX, list_dataY, list_dataR, max_len, bit_size, task, seed)

training_loader     = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True,drop_last= True, worker_init_fn = np.random.seed(0)) # Fix random seed của torch.utils.data.DataLoader , worker_init_fn = np.random.seed(0)
validation_loader   = torch.utils.data.DataLoader(validation_dataset, batch_size = batch_size, shuffle = False, drop_last= True)
test_loader         = torch.utils.data.DataLoader(test_dataset,  batch_size = batch_size, shuffle = False, drop_last= True)

##########################################################################################
# Load pretrain model(model multi_task) 
model_multi = Multi_task(bit_size, embedding_size, max_len, window_height, num_filter, dropout_rate).to(device)

# Loading model ứng với mỗi seed chia dữ liệu
model_multi = torch.load('/content/drive/MyDrive/Predict_task/Multi_task/pretrain_model/model_seed' + str(1) + '/Multi_model.pt')
model_multi.eval()

# Create Model (Single model)
model = Single_task(model_multi, bit_size, embedding_size, max_len, window_height, num_filter, dropout_rate, use_relation= True, use_multi_task= True).to(device)
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
    for batch_idx, (data, relation, labels) in enumerate(training_loader):
        data = data.to(device)
        relation = relation.type(torch.long).to(device)
        labels = labels.to(device)
        outputs = model(data, relation).view_as(labels)
        #------------------- 
        optimizer.zero_grad()
        #------------------- 
        loss = criteron(outputs, labels)  
        loss.backward()
        train_loss += loss.item()*len(data) #(loss.item is the average loss of training batch)
        optimizer.step() 
        #-------------------
        predicted = predict(outputs, len(data)).view_as(labels).to(device)
    #------------------- 
    print('====> Epoch: {} Average Train Loss: {:.4f}'.format(epoch, train_loss / len(training_loader.dataset)))
    train_loss = (train_loss / len(training_loader.dataset) )

    return train_loss

##########################################################################################
# Validation Function
def validate(epoch):
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for _, (data, relation, labels) in enumerate(validation_loader): 
            data = data.to(device)
            relation = relation.type(torch.long).to(device)
            labels = labels.to(device)
            outputs = model(data, relation).view_as(labels)
            loss = criteron(outputs, labels)
            validation_loss += loss.item()*len(data)
            predicted = predict(outputs, len(data)).view_as(labels).to(device)
    #------------------- 
    validation_loss = (validation_loss / len(validation_loader.dataset) )
    print('====> Average Validation Loss: {:.4f}'.format(validation_loss))
 
    return validation_loss
    
##########################################################################################
# Test Function
def test(epoch):
    model.eval()
    test_loss = 0
    pred_prob = []
    pred_class = []
    with torch.no_grad():
        for _, (data, relation, labels) in enumerate(test_loader): 
            data = data.to(device)
            relation = relation.type(torch.long).to(device)
            labels = labels.to(device)
            outputs = model(data, relation).view_as(labels)
            loss = criteron(outputs, labels)
            test_loss += loss.item()*len(data)
            pred_prob.append(outputs)
            predicted = predict(outputs, len(data)).view_as(labels).to(device)
            pred_class.append(predicted)
    #------------------- 
    test_loss = (test_loss / len(test_loader.dataset) )
    print('====> Average Test Loss: {:.4f}'.format(test_loss))

    return test_loss, pred_prob, pred_class

##########################################################################################                 
###                                  TRAINING MODEL                                    ###           
##########################################################################################
training_loss_list, validation_loss_list, test_loss_list = [], [], []
test_prob_pred = []
test_class_pred = []

#################################################################################
print("Training Model")
for epoch in range(1, 21):

    train_results = train(epoch)
    validation_results = validate(epoch)
  
    test_results = test(epoch)
    #------------------- 
    training_loss_list.append(train_results)
    validation_loss_list.append(validation_results)
    test_loss_list.append(test_results[0])
    #-------------------
    test_prob_pred.append(test_results[1])
    test_class_pred.append(test_results[2])

#################################################################################

# save_dir = '/content/drive/My Drive/Predict_task/save_result/random_seed_' + str(seed)+'/'

# # save ACC
# np.save(save_dir + 'Accuracy/used_pretrainMT_relation/Task' + str(task) + '/test_prob.npy', test_prob_pred) # thay doi model 
# np.save(save_dir + 'Accuracy/used_pretrainMT_relation/Task' + str(task) + '/test_class.npy', test_class_pred)

# # Save loss
# np.save(save_dir + 'Loss/used_pretrainMT_relation/Task' + str(task) + '/train_Loss.npy', training_loss_list)
# np.save(save_dir + 'Loss/used_pretrainMT_relation/Task' + str(task) + '/val_Loss.npy', validation_loss_list)


def evaluate(prob_list, class_list, best_epoch, test_set):
    # AUC (đánh giá độ tốt của mô hình nếu dữ liệu balance hoặc ko quá imbanace)
    roc_auc, fpr, tpr = Auc_score(prob_list, best_epoch, test_set)
    auc = np.round(roc_auc, 2) 
    # #Save AUC
    # np.save(save_dir + 'AUC/used_pretrainMT_relation/Task' + str(task) + '/tpr.npy', tpr)
    # np.save(save_dir + 'AUC/used_pretrainMT_relation/Task' + str(task) + '/fpr.npy', fpr)
   
    # MCC (đánh giá độ phân loại bài toán binary)
    mcc = np.round(Mcc_score(class_list, best_epoch, test_set), 2) 
    # Độ chính xác (ACC) (Đánh giá số lượng mẫu dự đoán đúng trên tổng mẫu), chỉ có ý nghĩa khi bài toán balance 
    acc = np.round(Acc_score(class_list, best_epoch, test_set), 2) 
    #------------------------------------------------
    # PR_AUC (đánh giá độ tốt của mô hình nếu dữ liệu imbalance)
    pr_auc, precision, recall = Pr_auc_score(prob_list, best_epoch, test_set)
    # # Save PR_AUC
    # np.save(save_dir + 'PR_AUC/used_pretrainMT_relation/Task' + str(task) + '/precision.npy', precision)
    # np.save(save_dir + 'PR_AUC/used_pretrainMT_relation/Task' + str(task) + '/recall.npy', recall)
    # F1_Score (đánh giá độ tách dữ dữ liệu -/+ đối với imbalance data)
    f1     = np.round(F1_score(class_list, best_epoch, test_set), 2)
    # Precision (đánh giá số lượng dự đoán đúng trên tổng dự đoán dương)
    pre    = np.round(Precision_score(class_list, best_epoch, test_set), 2)
    #------------------------------------------------
    spe    = np.round(Spe_score(prob_list, best_epoch, test_set), 2)
    # recall_score
    recall_score    = np.round(Recall_score(class_list, best_epoch, test_set), 2)
    return auc, acc, mcc, pr_auc, f1, pre, spe, recall_score

auc, acc, mcc, pr_auc, f1, pre, spe, rcs = evaluate(test_prob_pred, test_class_pred, np.argmin(validation_loss_list), test_loader)
print('seed: {}, task: {}, AUC: {}, ACC: {}, MCC: {}, PR_AUC: {}, F1: {}, PRE: {}, SPE: {}, Recall_score: {}'.format(seed,task, auc, acc, mcc, pr_auc, f1, pre, spe, rcs))
result = {}
result['Random_seed'] = seed
result['Task'] = task
result['Used relation'] = 'True' # Thay doi
result['Used preTrain'] = 'False' # Thay doi
result['AUC'] = auc
result['ACC'] = acc
result['MCC'] = mcc
result['PR_AUC'] = pr_auc
result['F1'] = f1
result['Pre'] = pre
result['Spe'] = spe

# save_result(result, save_dir= '/content/drive/My Drive/Predict_task/save_result/random_seed_' + str(seed))
