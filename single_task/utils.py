
# Import Python libraries
import csv, os
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
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, precision_score, recall_score
# Lưu tỷ lệ label(1, 0) trong mỗi bộ dataset(trainning dataset, validation dataset, test dataset)
def save_result(dictionary, save_dir="/content/drive/My Drive/Predict_task/dataset", filename='Result.csv'):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = os.path.join(save_dir, filename)
    if not (os.path.exists(path)):
        logfile = open(path, 'a')
        logwriter = csv.DictWriter(logfile, fieldnames=list(dictionary.keys()))
        logwriter.writeheader()
        logwriter = csv.DictWriter(logfile, fieldnames = dictionary.keys())
        logwriter.writerow(dictionary)
    else:
        logfile = open(path, 'a')
        logwriter = csv.DictWriter(logfile, fieldnames=dictionary.keys())
        logwriter.writerow(dictionary)
    logfile.close()

#---------------------------------------------------------------------
##########################################################################################
def predict(X, batch_size = 128):
    class_list = []
    for i in X:
        if i >= 0.5:
            class_list.append(1)
        else:
            class_list.append(0)
    class_list = torch.FloatTensor(class_list).view(batch_size, 1)
    return class_list

##########################################################################################
def Recall_score(prob_list, best_epoch, test_set):
    pred_list = []
    for batch_y in prob_list[best_epoch]:
        for i in batch_y:
            pred_list.append(i.detach().cpu().numpy())
    pred_list = np.array(pred_list).reshape(-1)
    
    real_list = []
    for _,_, batch_y in test_set:
        for y in batch_y:
            real_list.append(y.numpy())
    real_list = np.array(real_list) 
    
    acc = recall_score(real_list, pred_list)
    return acc
#############################################################################################
def Auc_score(prob_list, best_epoch, test_set):
    pred_list = []
    for batch_y in prob_list[best_epoch]:
        for i in batch_y:
            pred_list.append(i.detach().cpu().numpy())
    pred_list = np.array(pred_list)
    
    real_list = []
    for _,_, batch_y in test_set:
        for y in batch_y:
            real_list.append(y.numpy())
    real_list = np.array(real_list) 
    
    fpr, tpr, _ = metrics.roc_curve(real_list, pred_list)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc, fpr, tpr

##########################################################################################
def Spe_score(prob_list, best_epoch, test_set, sen_value=0.9):
    pred_list = []
    for batch_y in prob_list[best_epoch]:
        for i in batch_y:
            pred_list.append(i.detach().cpu().numpy())
    pred_list = np.array(pred_list)
    
    real_list = []
    for _,_, batch_y in test_set:
        for y in batch_y:
            real_list.append(y.numpy())
    real_list = np.array(real_list) 
    
    fpr, tpr, _ = metrics.roc_curve(real_list, pred_list)
    sen = np.round(tpr, 2)
    result = np. where(sen >= sen_value)
    cor_spe = 1 - fpr[result[0][0]]
    return cor_spe

##########################################################################################
def Mcc_score(prob_list, best_epoch, test_set):
    pred_list = []
    for batch_y in prob_list[best_epoch]:
        for i in batch_y:
            pred_list.append(i.detach().cpu().numpy())
    pred_list = np.array(pred_list).reshape(-1)
    
    real_list = []
    for _,_, batch_y in test_set:
        for y in batch_y:
            real_list.append(y.numpy())
    real_list = np.array(real_list) 
    
    mcc = matthews_corrcoef(real_list, pred_list)
    return mcc

##########################################################################################
def Acc_score(prob_list, best_epoch, test_set):
    pred_list = []
    for batch_y in prob_list[best_epoch]:
        for i in batch_y:
            pred_list.append(i.detach().cpu().numpy())
    pred_list = np.array(pred_list).reshape(-1)
    
    real_list = []
    for _,_, batch_y in test_set:
        for y in batch_y:
            real_list.append(y.numpy())
    real_list = np.array(real_list) 
    
    acc = accuracy_score(real_list, pred_list)
    return acc

##########################################################################################
def Pr_auc_score(prob_list, best_epoch, test_set):
    pred_list = []
    for batch_y in prob_list[best_epoch]:
        for i in batch_y:
            pred_list.append(i.detach().cpu().numpy())
    pred_list = np.array(pred_list)
    
    real_list = []
    for _,_, batch_y in test_set:
        for y in batch_y:
            real_list.append(y.numpy())
    real_list = np.array(real_list) 
    
    precision, recall, _ = metrics.precision_recall_curve(real_list, pred_list)
    pr_auc = metrics.auc(recall, precision)
    return pr_auc, precision, recall

##########################################################################################
def F1_score(prob_list, best_epoch, test_set):
    pred_list = []
    for batch_y in prob_list[best_epoch]:
        for i in batch_y:
            pred_list.append(i.detach().cpu().numpy())
    pred_list = np.array(pred_list)
    
    real_list = []
    for _,_, batch_y in test_set:
        for y in batch_y:
            real_list.append(y.numpy())
    real_list = np.array(real_list) 
    
    f1 = f1_score(real_list, pred_list)
    return f1

##########################################################################################
def Precision_score(prob_list, best_epoch, test_set, average='micro'):
    avg= average
    pred_list = []
    for batch_y in prob_list[best_epoch]:
        for i in batch_y:
            pred_list.append(i.detach().cpu().numpy())
    pred_list = np.array(pred_list)
    
    real_list = []
    for _,_, batch_y in test_set:
        for y in batch_y:
            real_list.append(y.numpy())
    real_list = np.array(real_list) 
    
    precision = precision_score(real_list, pred_list, average=avg)
    return precision
