# Import libraries
from sklearn.preprocessing import StandardScaler
from utils import *
from sklearn.cluster import KMeans
import numpy as np

#===========================================================================================
# Define cyp isoform
isoform = 'cyp1a2' # 'cyp2c9', 'cyp2c19', 'cyp2d6', 'cyp3a4'
index_vector  = np.load('./refined_data/index_vector/{}/data.npy'.format(isoform))
labels        = np.load('./refined_data/index_vector/{}/label.npy'.format(isoform))

#===========================================================================================
# Load dataset
data, smiles_multi, smiles_task, smiles_labels = load_data(isoform)

#===========================================================================================
# Normalizing data
scaler = StandardScaler()
data_normal = scaler.fit_transform(data)

#===========================================================================================
# Visualizing elbow plot to select suitable k 
elbow_plot(data_normal)

#Chose k = 6
n_clusters = 6

# lưu lại PCA để đánh giá
pca_plot(data_normal, n_clusters, isoform)

#===========================================================================================
# Define clusters using k-mean clustering algorithm
kmeans     = KMeans(n_clusters=6, random_state=0).fit(data_normal)
list_index = []
for i in range(n_clusters): 
    list_index.append(np.where(kmeans.labels_ == i)[0])

#===========================================================================================
# Split data
test_index  = sampling_from_clusters(list_index, len(kmeans.labels_),smiles_task, smiles_multi, None, 'test')
val_index   = sampling_from_clusters(list_index, len(kmeans.labels_),None, None, test_index, 'val')
train_index = []
for i in range(len(smiles_task)):
    # if  i not in list_index_test and i not in list_index_val:
    if  i not in test_index and i not in val_index:
        train_index.append(i)

train_data, train_label = index_vector[train_index], labels[train_index]
val_data,   val_label   = index_vector[val_index],   labels[val_index]
test_data,  test_label  = index_vector[test_index],  labels[test_index]

print("Training data: {}".format(train_data.shape))
print("Validation data: {}".format(val_data.shape))
print("Test data: {}".format(test_data.shape))

#===========================================================================================
# Save data
# np.save('./refined_data/splited_data/{}/train/data.npy'.format(type_cyp), train_data)
# np.save('./refined_data/splited_data/{}/train/label.npy'.format(type_cyp),train_label)
# np.save('./refined_data/splited_data/{}/val/data.npy'.format(type_cyp), val_data)
# np.save('./refined_data/splited_data/{}/val/label.npy'.format(type_cyp), val_label)
# np.save('./refined_data/splited_data/{}/test/data.npy'.format(type_cyp), test_data)
# np.save('./refined_data/splited_data/{}/test/label.npy'.format(type_cyp), test_label)
