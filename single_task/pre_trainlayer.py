# Import Python libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import random

##########################################################################################
class Multi_task(nn.Module):
    torch.manual_seed(1)
    def __init__(self, bit_size = 2048, embedding_size = 200, max_length = 200, window_height = 5, num_filter = 4096, dropout_rate=0):
        super(Multi_task, self).__init__()
        np.random.seed(1)
        random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Network Architecture
        self.embeddings     = nn.Embedding(bit_size + 1, embedding_size, padding_idx = 0)
        self.bnorm_emb      = nn.BatchNorm2d(num_features = 1)   
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
        # Task 1
        self.linear20        = nn.Linear(512, 256)
        self.drop20          = nn.Dropout(p=dropout_rate)
        self.linear30        = nn.Linear(256, 1)
        # Task 2
        self.linear21        = nn.Linear(512, 256)
        self.drop21          = nn.Dropout(p=dropout_rate)
        self.linear31        = nn.Linear(256, 1)
        # Task 3
        self.linear22        = nn.Linear(512, 256)
        self.drop22          = nn.Dropout(p=dropout_rate)
        self.linear32        = nn.Linear(256, 1)
        # Task 4
        self.linear23        = nn.Linear(512, 256)
        self.drop23          = nn.Dropout(p=dropout_rate)
        self.linear33        = nn.Linear(256, 1)
        # Task 
        self.linear24        = nn.Linear(512, 256)
        self.drop24         = nn.Dropout(p=dropout_rate)
        self.linear34        = nn.Linear(256, 1)

        self.bnorm_l2       = nn.BatchNorm1d(num_features = 1)
        self.activate_l2    = nn.LeakyReLU()
        self.activate_l3    = nn.Sigmoid()
        #####################
        #Variables
        self.embedded_size  = embedding_size
        self.max_length     = max_length
        self.pooled_size    = num_filter
        
    def forward(self, x):
        #Share Layer --------------------------------------------------
        #Embedding
        embeds              = self.embeddings(x).view(-1, 1, self.max_length, self.embedded_size)
        embeds              = self.bnorm_emb(embeds) # torch.Size([128, 1, 200, 200])
        # Convolution
        output1             = self.conv1(embeds) # kernel_size = (5, 200))
        output1             = self.bnorm_conv1(output1)
        output1             = self.activate_conv1(output1) # torch.Size([128, 4096, 196, 1])
        # Maxpooling
        output1             = self.pool_conv1(output1).view(-1, 1, self.pooled_size) # torch.Size([128, 1, 4096])
        #Linear
        output2             = self.linear1(output1)
        output2             = self.bnorm_l1(output2) 
        output2             = self.activate_l1(output2) 
        output2             = self.drop1(output2)

        #Task Layer ---------------------------------------------------
        # Task 1
        task1               = self.linear20(output2)
        task1               = self.bnorm_l2(task1) 
        task1               = self.activate_l2(task1)
        task1               = self.drop20(task1)            
        task1               = self.linear30(task1)
        task1               = self.activate_l3(task1)

        # Task 2
        task2               = self.linear21(output2)
        task2               = self.bnorm_l2(task2) 
        task2               = self.activate_l2(task2)
        task2               = self.drop21(task2)            
        task2               = self.linear31(task2)
        task2               = self.activate_l3(task2)

        # Task 3
        task3               = self.linear22(output2)
        task3               = self.bnorm_l2(task3) 
        task3               = self.activate_l2(task3)
        task3               = self.drop22(task3)            
        task3               = self.linear32(task3)
        task3               = self.activate_l3(task3)
        # Task 4
        task4               = self.linear23(output2)
        task4               = self.bnorm_l2(task4) 
        task4               = self.activate_l2(task4)
        task4               = self.drop23(task4)            
        task4               = self.linear33(task4)
        task4               = self.activate_l3(task4)
        # Task 5
        task5               = self.linear24(output2)
        task5               = self.bnorm_l2(task5) 
        task5               = self.activate_l2(task5)
        task5               = self.drop24(task5)            
        task5               = self.linear34(task5)
        task5               = self.activate_l3(task5)
        return output2, task1, task2 , task3, task4, task5