# Import Library
from training_func import *

#==========================================================================================#                 
#                                 BENCHMARKING MODEL                                       #           
#==========================================================================================#
# Define parameter
isoform_list = ['cyp1a2', 'cyp2c9', 'cyp2c19', 'cyp2d6', 'cyp3a4']

# Training
knn_benchmark(isoform_list)
svm_benchmark(isoform_list)
rf_benchmark(isoform_list)