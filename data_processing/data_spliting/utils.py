# Import libraries
import matplotlib.pyplot as plt 
from rdkit.Chem import AllChem
from rdkit import Chem
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import random 

#===========================================================================================
# Extract Morgan fingerprint 
def smiles2fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol != None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits = 2048)
        return fp
    else:
        print("smiles {} is invalid !".format(smiles))
        return None

#===========================================================================================
# Convert Morgan fingerprint vectors to index vectors  
def fp2idxvec(x, max_len = 200, bit_size = 2048):
    data_x = []
    for i in range(len(x)):
        fp = [0] * max_len
        n_ones = 0
        for j in range(bit_size):
            if x[i][j] == 1:
                fp[n_ones] = j+1
                n_ones += 1
        data_x.append(fp)
    data_x = np.array(data_x, dtype=np.int32)
    return data_x

#===========================================================================================
# Get sample from each cluster to create train, val, test sets
def sampling_from_clusters(list_index, data_length, smiles_task= None, smiles_multi= None, test_index= None, sample = 'test', n_clusters = 6, seed = 0):
    random.seed(seed) 
    count = 0
    index_sample = []
    #--------------------
    if sample == 'test':
        n_sameple = 1000
    if sample == 'val':
        n_sameple = 2000
    #--------------------
    for k in range(n_clusters):
        # số lượng mẫu tính theo tỷ lệ
        ratio = int(len(list_index[k])/data_length * n_sameple)
        if k == n_clusters - 1:
            ratio = n_sameple - count
        i = 0
        while i < ratio:
            # random cụm 0
            smiles_idx = random.choice(list_index[k])
            if sample == 'test' and smiles_task[smiles_idx] not in smiles_multi and smiles_idx not in index_sample:
                index_sample.append(smiles_idx)
                i +=1
            if sample == 'val' and smiles_idx not in test_index and smiles_idx not in index_sample:
                index_sample.append(smiles_idx)
                i +=1
        count += ratio
    return index_sample

#===========================================================================================
def load_data(isoform):
    isoform_ = isoform
    path1    = './data/CYP_md/md_{}.csv'.format(isoform_) 
    path2    = '/source_cyp/data_processing/data_shared_weight_extracting/data/multi_dataCYP.csv'
    df_path1 = pd.read_csv(path1) # Task data
    df_path2 = pd.read_csv(path2) # Multi data
    #--------------------
    # Delete colums contain nan
    df_path1 = df_path1.drop(columns=['MaxAbsPartialCharge', 'MaxPartialCharge', 'MinAbsPartialCharge', 'MinPartialCharge']) 
    #--------------------
    # Loai bo 3 cot dau chua thong tin ve smile, label , ID
    vector = np.array(df_path1)
    data   = vector[::,4:]
    #--------------------
    smiles_multi = df_path2['Cano_Smile'].tolist() # List SMILES of multi-data
    smiles_task  = df_path1['Cano_Smile'].tolist() # List SMILES of each task
    labels       = df_path1['Label'].tolist() # Label of SMILES
    #--------------------
    return data, smiles_multi, smiles_task, labels

#===========================================================================================
# Show plot of turning k in algorithm k-mean cluster 
def elbow_plot(data_normal):
    cost =[]
    for i in range(1, 11):
        KM = KMeans(n_clusters = i, max_iter = 1000)
        KM.fit(data_normal)
        #--------------------
        # calculates squared error for the clustered points
        cost.append(KM.inertia_)	
    # Plot the cost against k values
    plt.plot(range(1, 11), cost, color ='g', linewidth ='3')
    plt.xlabel("Value of K")
    plt.ylabel("Sqaured Error (Cost)")
    plt.show() # clear the plot

#===========================================================================================
# Show clusters in PCA
def pca_plot(data_normal, n_clusters, cyp_action):
    pca = PCA(n_components= 192)
    principalComponents = pca.fit_transform(data_normal)
    PCA_components = pd.DataFrame(principalComponents)

    model = KMeans(n_clusters= n_clusters)
    model.fit(PCA_components.iloc[:,:2])

    labels = model.predict(PCA_components.iloc[:,:2])
    plt.scatter(PCA_components[0], PCA_components[1], c=labels)
    plt.savefig('./images/PCA/PCA' + cyp_action + '.pdf')
    plt.show()