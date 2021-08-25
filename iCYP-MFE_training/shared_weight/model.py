# Import libraries
import torch
import torch.nn as nn
import numpy as np
import random

#===================================================================================================================================
class multi_task(nn.Module):
    torch.manual_seed(1)
    def __init__(self, bit_size = 2048, embedding_size = 200, max_length = 200, window_height = 5, num_filter = 4096, dropout_rate=0):
        super(multi_task, self).__init__()
        np.random.seed(1)
        random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        #====================
        # Network Architecture
        self.embeddings     = nn.Embedding(bit_size + 1, embedding_size, padding_idx = 0)
        self.bnorm_emb      = nn.BatchNorm2d(num_features = 1)   

        #====================
        self.conv1          = nn.Conv2d(in_channels = 1, out_channels = num_filter, kernel_size = (window_height, embedding_size)) # (1024, 96, 1)
        self.bnorm_conv1    = nn.BatchNorm2d(num_features = num_filter)
        self.activate_conv1 = nn.LeakyReLU()
        self.pool_conv1     = nn.MaxPool2d(kernel_size = (max_length - window_height + 1, 1)) # Output of layer max-pooling is outout of share layer
 
        #====================
        # Task 1
        self.linear1_task1        = nn.Linear(num_filter, 512)
        self.bnorm_task1          = nn.BatchNorm1d(num_features = 1)
        self.activate_task1       = nn.LeakyReLU()
        self.drop1_task_1         = nn.Dropout(p=dropout_rate)
        self.linear2_task1        = nn.Linear(512, 256)
        self.drop2_task1          = nn.Dropout(p=dropout_rate)
        self.linear3_task1        = nn.Linear(256, 1)
        #--------------------  
        # Task 2
        self.linear1_task2        = nn.Linear(num_filter, 512)
        self.bnorm_task2          = nn.BatchNorm1d(num_features = 1)
        self.activate_task2       = nn.LeakyReLU()
        self.drop1_task_2         = nn.Dropout(p=dropout_rate)
        self.linear2_task2        = nn.Linear(512, 256)
        self.drop2_task2          = nn.Dropout(p=dropout_rate)
        self.linear3_task2        = nn.Linear(256, 1)
        #--------------------
        # Task 3
        self.linear1_task3        = nn.Linear(num_filter, 512)
        self.bnorm_task3          = nn.BatchNorm1d(num_features = 1)
        self.activate_task3       = nn.LeakyReLU()
        self.drop1_task_3         = nn.Dropout(p=dropout_rate)
        self.linear2_task3        = nn.Linear(512, 256)
        self.drop2_task3          = nn.Dropout(p=dropout_rate)
        self.linear3_task3        = nn.Linear(256, 1)
        #--------------------
        # Task 4
        self.linear1_task4        = nn.Linear(num_filter, 512)
        self.bnorm_task4          = nn.BatchNorm1d(num_features = 1)
        self.activate_task4       = nn.LeakyReLU()
        self.drop1_task_4         = nn.Dropout(p=dropout_rate)
        self.linear2_task4        = nn.Linear(512, 256)
        self.drop2_task4          = nn.Dropout(p=dropout_rate)
        self.linear3_task4        = nn.Linear(256, 1)
        #--------------------
        # Task 5
        self.linear1_task5        = nn.Linear(num_filter, 512)
        self.bnorm_task5          = nn.BatchNorm1d(num_features = 1)
        self.activate_task5       = nn.LeakyReLU()
        self.drop1_task_5         = nn.Dropout(p=dropout_rate)
        self.linear2_task5        = nn.Linear(512, 256)
        self.drop2_task5          = nn.Dropout(p=dropout_rate)
        self.linear3_task5        = nn.Linear(256, 1)
        #--------------------
        self.sigmoid              = nn.Sigmoid()
        #--------------------
        # Variables
        self.embedded_size        = embedding_size
        self.max_length           = max_length
        self.pooled_size          = num_filter
        
    def forward(self, x):
        #====================
        # Shared-weight 
        #====================
        # Embedding
        embeds              = self.embeddings(x).view(-1, 1, self.max_length, self.embedded_size)
        embeds              = self.bnorm_emb(embeds) # torch.Size([128, 1, 200, 200])
        # Convolution
        out_conv            = self.conv1(embeds) # kernel_size = (5, 200))
        out_conv            = self.bnorm_conv1(out_conv)
        out_conv            = self.activate_conv1(out_conv) # torch.Size([128, 4096, 196, 1])
        # Max-pooling
        out_share_layer     = self.pool_conv1(out_conv).view(-1, 1, self.pooled_size) # torch.Size([128, 1, 4096])
        #====================    
        # Task layers
        #====================
        # Task 1
        task1               = self.linear1_task1(out_share_layer)
        task1               = self.bnorm_task1(task1)
        task1               = self.activate_task1(task1)
        task1               = self.drop1_task_1(task1)            
        task1               = self.linear2_task1(task1)
        task1               = self.drop2_task1(task1)
        task1               = self.linear3_task1(task1)
        out_task1           = self.sigmoid(task1).view(-1)
        #--------------------
        # Task 2
        task2               = self.linear1_task2(out_share_layer)
        task2               = self.bnorm_task2(task2)
        task2               = self.activate_task2(task2)
        task2               = self.drop1_task_2(task2)            
        task2               = self.linear2_task2(task2)
        task2               = self.drop2_task2(task2)
        task2               = self.linear3_task2(task2)
        out_task2           = self.sigmoid(task2).view(-1)
        #--------------------
        # Task 3
        task3               = self.linear1_task3(out_share_layer)
        task3               = self.bnorm_task3(task3)
        task3               = self.activate_task3(task3)
        task3               = self.drop1_task_3(task3)            
        task3               = self.linear2_task3(task3)
        task3               = self.drop2_task3(task3)
        task3               = self.linear3_task3(task3)
        out_task3           = self.sigmoid(task3).view(-1)
        #--------------------
        # Task 4
        task4               = self.linear1_task4(out_share_layer)
        task4               = self.bnorm_task4(task4)
        task4               = self.activate_task4(task4)
        task4               = self.drop1_task_4(task4)            
        task4               = self.linear2_task4(task4)
        task4               = self.drop2_task4(task4)
        task4               = self.linear3_task4(task4)
        out_task4           = self.sigmoid(task4).view(-1)
        #--------------------
        # Task 5
        task5               = self.linear1_task5(out_share_layer)
        task5               = self.bnorm_task5(task5)
        task5               = self.activate_task5(task5)
        task5               = self.drop1_task_5(task5)            
        task5               = self.linear2_task5(task5)
        task5               = self.drop2_task5(task5)
        task5               = self.linear3_task5(task5)
        out_task5           = self.sigmoid(task5).view(-1)
        #====================
        return out_share_layer, (out_task1, out_task2 , out_task3, out_task4, out_task5)