# Import libraries
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from sklearn.model_selection import train_test_split

#===========================================================================================
# Extract Morgan fingerprint 
def smiles2fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol != None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits = 2048)
        return fp
    else:
        print("SMILES {} is invalid !".format(smiles))
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
# Split data to training and validation sets for shared-weight networks
def split_data(dataX, list_dataY, seed=1):
    X_train, X_val, Y_train0, Y_val0 = train_test_split(dataX, list_dataY[0], test_size=0.1, random_state=seed)
    X_train, X_val, Y_train1, Y_val1 = train_test_split(dataX, list_dataY[1], test_size=0.1, random_state=seed)
    X_train, X_val, Y_train2, Y_val2 = train_test_split(dataX, list_dataY[2], test_size=0.1, random_state=seed)
    X_train, X_val, Y_train3, Y_val3 = train_test_split(dataX, list_dataY[3], test_size=0.1, random_state=seed)
    X_train, X_val, Y_train4, Y_val4 = train_test_split(dataX, list_dataY[4], test_size=0.1, random_state=seed)
    #--------------------
    Y_train = [Y_train0, Y_train1, Y_train2, Y_train3, Y_train4]
    Y_val  = [Y_val0, Y_val1, Y_val2, Y_val3, Y_val4]  
    #--------------------
    return X_train, Y_train, X_val, Y_val
