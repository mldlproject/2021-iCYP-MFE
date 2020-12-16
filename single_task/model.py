# Import Python libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import random

##########################################################################################
class Single_task(nn.Module):
    torch.manual_seed(1)
    def __init__(self, multi_task, bit_size = 1024, embedding_size = 100, max_length = 100, window_height = 5, num_filter = 1024, dropout_rate=0, use_relation = True, use_multi_task = True):
        super(Single_task, self).__init__()
        np.random.seed(1)
        random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        #Network Architecture
        self.embeddings     = nn.Embedding(bit_size + 1, embedding_size, padding_idx = 0) #(100, 100)
        self.bnorm_emb      = nn.BatchNorm2d(num_features = 1)   


        self.embedding_relation = nn.Embedding(256, 4, padding_idx = 0)
        self.linear_relation    = nn.Linear(16, 64)
        self.linear_relation2   = nn.Linear(64, 128)
        #####################
        self.conv1          = nn.Conv2d(in_channels = 1, out_channels = num_filter, kernel_size = (window_height, embedding_size)) # (1024, 96, 1)
        self.bnorm_conv1    = nn.BatchNorm2d(num_features = num_filter)
        self.activate_conv1 = nn.LeakyReLU()
        self.pool_conv1     = nn.MaxPool2d(kernel_size = (max_length - window_height + 1, 1))

        #####################
        self.linear1        = nn.Linear(num_filter, 512)
        self.bnorm_l1       = nn.BatchNorm1d(num_features = 1)
        self.activate_l1    = nn.LeakyReLU()
        self.drop1          = nn.Dropout(p=dropout_rate)

        #####################
        self.linear2        = nn.Linear(512, 128)
        self.bnorm_l2       = nn.BatchNorm1d(num_features = 1)
        self.activate_l2    = nn.LeakyReLU()
        self.drop2          = nn.Dropout(p=dropout_rate)
        #####################

        self.linear31        = nn.Linear(256, 1)# Sử dụng nếu model có relation vector: used_relation = True
        self.linear32        = nn.Linear(128, 1)
        self.activate_l3    = nn.Sigmoid()
        #####################
        #Variables
        self.embedded_size  = embedding_size
        self.max_length     = max_length
        self.pooled_size    = num_filter
        self.use_relation   = use_relation
        self.use_multi_task = use_multi_task
        self.multi_task     = multi_task # Model pretrain
    def forward(self, x, relation):
        # Embedding
        embeds              = self.embeddings(x).view(-1, 1, self.max_length, self.embedded_size)
        embeds              = self.bnorm_emb(embeds)
        vector_embedd       = self.embedding_relation(relation).view(128, -1)

        # Embedding vector relation
        vector_relation     = self.linear_relation(vector_embedd)
        vector_relation     = self.linear_relation2(vector_relation)
        
        #####################
        if self.use_multi_task == False:
            output1             = self.conv1(embeds)
            output1             = self.bnorm_conv1(output1)
            output1             = self.activate_conv1(output1)
            output1             = self.pool_conv1(output1).view(-1, 1, self.pooled_size) 
            # print(output1.shape)
            #output1             = self.drop_conv1(output1)
            #####################
            output2             = self.linear1(output1)
            # print(output2.shape) 
            output2             = self.bnorm_l1(output2) 
            output2             = self.activate_l1(output2) 
            output2             = self.drop1(output2) 
            #####################
        else:
            output2, task1, task2, task3, task4, task5 = self.multi_task(x) # Using pretrain layer
        output3             = self.linear2(output2)
        output3             = self.bnorm_l2(output3) 
        output3             = self.activate_l2(output3)

        if self.use_relation == True: # Sử dụng vector relation 
            output3             = self.drop2(output3).view(128, -1)    
            output4             = torch.cat([output3,vector_relation], dim = -1) # Concatinate 
            output4             = self.linear31(output4)
        else:
            output3             = self.drop2(output3)
            output4             = self.linear32(output3)
        output4             = self.activate_l3(output4)
        return output4
##########################################################################################
