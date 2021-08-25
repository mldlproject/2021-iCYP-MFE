# Import Python libraries
import torch
import torch.nn as nn
import numpy as np
import random

#===================================================================================================================================
class single_task(nn.Module):
    def __init__(self, model_multi, bit_size = 1024, embedding_size = 100, max_length = 100, window_height = 5, num_filter = 1024, dropout_rate=0, share_layer = True):
        super(single_task, self).__init__()
        np.random.seed(1)
        random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        #====================
        #Network Architecture
        self.embeddings          = nn.Embedding(bit_size + 1, embedding_size, padding_idx = 0)
        self.bnorm_emb           = nn.BatchNorm2d(num_features = 1)   
        #====================
        self.conv1               = nn.Conv2d(in_channels = 1, out_channels = num_filter, kernel_size = (window_height, embedding_size))
        self.bnorm_conv1         = nn.BatchNorm2d(num_features = num_filter)
        self.activate_conv1      = nn.LeakyReLU()
        self.pool_conv1          = nn.MaxPool2d(kernel_size = (max_length - window_height + 1, 1))
        #====================
        # Task 
        self.linear1_task        = nn.Linear(num_filter, 512)
        self.bnorm_task          = nn.BatchNorm1d(num_features = 1)
        self.activate_task       = nn.LeakyReLU()
        self.drop1_task          = nn.Dropout(p=dropout_rate)
        self.linear2_task        = nn.Linear(512, 256)
        self.drop2_task          = nn.Dropout(p=dropout_rate)
        self.linear3_task        = nn.Linear(256, 1)
        self.sigmoid             = nn.Sigmoid()
        #====================
        #Variables
        self.embedded_size       = embedding_size
        self.max_length          = max_length
        self.pooled_size         = num_filter
        self.share_layer         = share_layer
        self.model_multi         = model_multi
        
    def forward(self, x):
        #====================
        # Model
        if self.share_layer == False: 
            #Embedding
            embeds               = self.embeddings(x).view(-1, 1, self.max_length, self.embedded_size)
            embeds               = self.bnorm_emb(embeds) # torch.Size([batch, 1, 200, 200])
            # Convolution
            out_conv             = self.conv1(embeds) # kernel_size = (5, 200))
            out_conv             = self.bnorm_conv1(out_conv)
            out_conv             = self.activate_conv1(out_conv) # torch.Size([batch, 2048, 196, 1])
            # Maxpooling
            out_maxpooling       = self.pool_conv1(out_conv).view(-1, 1, self.pooled_size) # torch.Size([batch, 1, 2048])
            # FC1
            task                 = self.linear1_task(out_maxpooling)
            task                 = self.bnorm_task(task)
            task                 = self.activate_task(task)
            task                 = self.drop1_task(task)
            # FC2
            task                 = self.linear2_task(task)
            task                 = self.drop2_task(task)
            # FC3
            task                 = self.linear3_task(task)
            output               = self.sigmoid(task).view(-1)
        #====================
        # Model using pretrain share layer 
        else:
            out_share_layer, _   = self.model_multi(x) # out_of_sharelayer
            # FC1
            task                 = self.linear1_task(out_share_layer)
            task                 = self.bnorm_task(task)
            task                 = self.activate_task(task)
            task                 = self.drop1_task(task)
            # FC2
            task                 = self.linear2_task(task)
            task                 = self.drop2_task(task)
            # FC3
            task                 = self.linear3_task(task)
            output               = self.sigmoid(task).view(-1)
        #====================
        return output

