# Process Data
'''
+ Xử lý dataset: Trong bộ dữ liệu gồm chuỗi smile của mỗi chất và label của mỗi chất tương ứng với mỗi task
+ Dataset for multi_task model: bộ dữ liệu gồm 5.1k chất chứa đủ nhãn của 5 lablel

==> Mã hóa Smile -> vector index 

'''
import os, sys
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
import pandas as pd

##########################################################################################
# Load
df = pd.read_csv('/content/drive/MyDrive/Predict_task/Multi_task/Data/data_multi.csv')

dataX = []
dataY1, dataY2, dataY3, dataY4, dataY5 = [],[],[],[],[]
dataY = [dataY1, dataY2, dataY3, dataY4, dataY5]
none_index = []
list_class = ["cyp2c19_action", "cyp2d6_action", "cyp3a4_action", "cyp1a2_action", "cyp2c9_action"]
for i in range(0, len(df)): # Mã hóa smile của mỗi chất về dạng vector index
    smiles_list = df['Smile']
    smiles = smiles_list[i]
    try:
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits = 2048)
        fp = np.array(fp)
        dataX.append(fp)
    except:
            print("Smile không hợp lệ !")
for index in range(len(list_class)): # Lưu label của mỗi chất ứng mỗi task
    
    vallist = df[list_class[index]]
    for i in range(0, len(df)):
        smiles_list = df['Smile']
        smiles = smiles_list[i]
        try:
            mol = Chem.MolFromSmiles(smiles)
            if vallist[i] == 1:
                dataY[index].append(1)
            elif vallist[i] == 0:
                dataY[index].append(0)
        except:
            continue
##########################################################################################
dataX = np.array(dataX)
print(dataX.shape)
dataY1, dataY2, dataY3, dataY4, dataY5 = np.array(dataY1), np.array(dataY2), np.array(dataY3), np.array(dataY4), np.array(dataY5)
print("labels task 1", dataY1.shape)
list_dataY = [dataY1, dataY2, dataY3, dataY4, dataY5]
np.save('/content/drive/My Drive/Predict_task/Multi_task/Data/dataX.npy', dataX)
np.save('/content/drive/My Drive/Predict_task/Multi_task/Data/dataY.npy', list_dataY)