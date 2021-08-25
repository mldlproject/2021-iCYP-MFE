# Import libraries
import pandas as pd
import numpy as np
from utils import *

#===========================================================================================
# Load dataset
list_cyp = ["cyp2c19_action", "cyp2d6_action", "cyp3a4_action", "cyp1a2_action", "cyp2c9_action"]
df       = pd.read_csv('./data/multi_dataCYP.csv')

#===========================================================================================
# Storing labels of compounds in each task (cyp isoform) are as lists
list_label_task = [df[cyp_type].tolist() for cyp_type in list_cyp] # Labels of compounds in each task (cyp isoform) are stored as lists
dataX = []
dataY = [[], [], [], [], []]

#===========================================================================================
# SMILES list
smiles_list = df['Cano_Smile'].tolist()

#===========================================================================================
for i in range(0, len(smiles_list)):
    # Check valid smiles 
    smiles = smiles_list[i]
    #--------------------
    # Converting to fingerprint vectors  
    fp = np.array(smiles2fp(smiles))
    dataX.append(fp)
    #--------------------
    # Saving labels of compounds in each task
    for idx in range(len(list_label_task)):
        task_cyp = list_label_task[idx]
        if task_cyp[i] == 1:
            dataY[idx].append(1)
        elif task_cyp[i] == 0:
            dataY[idx].append(0)
            
#===========================================================================================
# Vector fb to vector index
index_vector = fp2idxvec(dataX)

# Split data into training and validation sets for shared-weight networks
train_data, train_label, val_data, val_label = split_data(index_vector, dataY)
# print(train_data.shape)
# print(val_data.shape)

#===========================================================================================
# save_data
# np.save('./refined_data/shared_weight_val/multi_refined_data.npy',val_data)
# np.save('./refined_data/shared_weight_val/multi_refined_labels.npy', val_label)

# np.save('./refined_data/shared_weight_train/multi_refined_data.npy',train_data)
# np.save('./refined_data/shared_weight_train/multi_refined_labels.npy', train_label)




