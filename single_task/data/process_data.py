import os, sys
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
import pandas as pd
'''
Tính toán cho từng task
    + Chuyển dữ liệu từ smile quan fingerprint
    + Tính toán vector quan hệ giữa các task 
    + lưu label của mỗi chất ứng với mỗi task
'''
# Load
df = pd.read_csv('/content/drive/MyDrive/Predict_task/Single_task/data/CYP450_AID_1851_datatable_all_SMILES.csv')

dataX1, dataX2, dataX3, dataX4, dataX5 = [],[],[],[],[]
dataX = [dataX1, dataX2, dataX3, dataX4, dataX5]
dataY1, dataY2, dataY3, dataY4, dataY5 = [],[],[],[],[]
dataY = [dataY1, dataY2, dataY3, dataY4, dataY5]
dataR1, dataR2, dataR3, dataR4, dataR5 = [],[],[],[],[]
dataR = [dataR1, dataR2, dataR3, dataR4, dataR5]

none_index = []
list_class = ['cyp2c19_action', "cyp2d6_action", "cyp3a4_action", "cyp1a2_action", "cyp2c9_action"]
list_warning = ['cyp2c19_warning', "cyp2d6_warning", "cyp3a4_warning", "cyp1a2_warning", "cyp2c9_warning"]
for index in range(len(list_class)):
    for i in range(0, len(df)):
        smiles_list = df['Smile']
        smiles = smiles_list[i]
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol == None:
                none_index.append(i)
            else:    
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits = 2048)
                fp = np.array(fp)
                vallist = df[list_class[index]]
                warning = df[list_warning[index]]
                if vallist[i] == 1: #action == 1 
                    dataX[index].append(fp)
                    dataY[index].append(1)
                    vector_relation = []
                    list_class_ = [x for x in list_class if x != list_class[index]] # List sau khi xóa task được tình vector quan hệ 
                    list_warning_ = [x for x in list_warning if x != list_warning[index]]

                    for idx in range(len(list_class_)): # 
                        val = df[list_class_[idx]]
                        war = df[list_warning_[idx]]
                        if war[i] == 1: # Chất đó không tồn tại trong các task đang được xét
                            vector_relation.append(5) # Định danh 5 đại diện cho chất không tồn tại
                        elif val[i] == 1:
                            vector_relation.append(idx+1) # Định danh của mỗi ta
                    while len(vector_relation) < 4: # add padding
                        vector_relation.append(0)
                    dataR[index].append(vector_relation)
                        
                elif vallist[i] == 0 and warning[i] == 0:
                    dataX[index].append(fp)
                    dataY[index].append(0)
                    vector_relation = []
                    list_class_ = [x for x in list_class if x != list_class[index]] # List sau khi xóa task được tình vector quan hệ 
                    list_warning_ = [x for x in list_warning if x != list_warning[index]]

                    for idx in range(len(list_class_)): # 
                        val = df[list_class_[idx]]
                        war = df[list_warning_[idx]]
                        if war[i] == 1:
                            vector_relation.append(5)
                        elif val[i] == 1:
                            vector_relation.append(idx+1)
                    while len(vector_relation) < 4: # padding
                        vector_relation.append(0)
                    dataR[index].append(vector_relation)
        except:
              print("something is wrong")

##########################################################################################
# sys.exit()

dataX1 = np.array(dataX1) # (13427, 2048)
dataX2 = np.array(dataX2) # (13896, 2048)
dataX3 = np.array(dataX3) # (12997, 2048)
dataX4 = np.array(dataX4) # (13239, 2048)
dataX5 = np.array(dataX5) # (12881, 2048)

print("data task 1:", dataX1.shape)
print("data task 2:", dataX2.shape)
print("data task 3:", dataX3.shape)
print("data task 4:", dataX4.shape)
print("data task 5:", dataX5.shape)
dataR1, dataR2, dataR3, dataR4, dataR5 = np.array(dataR1).reshape(-1,4), np.array(dataR2).reshape(-1,4), np.array(dataR3).reshape(-1,4), np.array(dataR4).reshape(-1,4), np.array(dataR5).reshape(-1,4)
dataY1, dataY2, dataY3, dataY4, dataY5 = np.array(dataY1), np.array(dataY2), np.array(dataY3), np.array(dataY4), np.array(dataY5)

# Dữ liệu của 5 task
list_dataX = [dataX1, dataX2, dataX3, dataX4, dataX5]
list_dataR = [dataR1, dataR2, dataR3, dataR4, dataR5]
list_dataY = [dataY1, dataY2, dataY3, dataY4, dataY5]

#Save
np.save("/content/drive/MyDrive/Predict_task/Single_task/data/dataX.npy", list_dataX)
np.save("/content/drive/MyDrive/Predict_task/Single_task/data/dataR.npy", list_dataR)
np.save("/content/drive/MyDrive/Predict_task/Single_task/data/dataY.npy", list_dataY)