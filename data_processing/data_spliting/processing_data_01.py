# Import libraries
import numpy as np
import pandas as pd
from utils import *

#===========================================================================================
# Load data
df_1a2  = pd.read_csv('./data/CYP_md/md_cyp1a2.csv')
df_2c9  = pd.read_csv('./data/CYP_md/md_cyp2c9.csv')
df_2c19 = pd.read_csv('./data/CYP_md/md_cyp2c19.csv')
df_2d6  = pd.read_csv('./data/CYP_md/md_cyp2d6.csv')
df_3a4  = pd.read_csv('./data/CYP_md/md_cyp3a4.csv')

list_task     = [df_1a2, df_2c9, df_2c19, df_2d6, df_3a4]
list_class    = ['cyp1a2_action', 'cyp2c19_action', 'cyp2c9_action', 'cyp2d6_action', 'cyp3a4_action']
list_type_cyp = ['cyp1a2', 'cyp2c19', 'cyp2c9', 'cyp2d6', 'cyp3a4']

#===========================================================================================
# Convert data of each task into index vectors 
for index in range(len(list_class)):
    dataX, dataY = [] ,[]
    for i in range(0, len(list_task[index])):
        df = list_task[index]
        smiles_list = df['Cano_Smile']
        smiles = smiles_list[i]
        #--------------------
        # SMILES to fingerprint vector
        fp = np.array(smiles2fp(smiles))
        dataX.append(fp)  
        labels = df['Label']
        #--------------------
        # Label
        if labels[i] == 1: #action == 1 
            dataY.append(1)
        else:
            dataY.append(0)
    #--------------------
    # Converting into index vectors
    index_vector = fp2idxvec(dataX)
    dataY = np.array(dataY)
    #--------------------
    print("Data for task {}".format(list_type_cyp[index]))
    print(index_vector.shape)
    print(dataY.shape)
    print('====================================================================')
    #--------------------
    np.save("./refined_data/index_vector/{}/data.npy".format(list_type_cyp[index]), index_vector)
    np.save("./refined_data/index_vector/{}/label.npy".format(list_type_cyp[index]), dataY)
