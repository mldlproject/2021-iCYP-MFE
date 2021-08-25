# Import libraries
import time
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import make_pipeline
from utils import printPerformance
from sklearn.metrics import roc_auc_score

#===========================================================================================
# Remove Low-variance Features
from sklearn.feature_selection import VarianceThreshold
threshold = (.95 * (1 - .95))

# Normalize Features
from sklearn.preprocessing import MinMaxScaler

#==========================================================================================#                 
#                                 BENCHMARKING KNN MODEL                                   #           
#==========================================================================================#
def knn_benchmark(isoform_list):
    start = time.time()
    columns_= ['ACC', 'BA', 'ROC-AUC', 'PR-AUC', 'MCC', 'SN', 'SP', 'PR', 'F1', 'CK']
    df = pd.DataFrame(columns_)
    isoform_list_ = isoform_list
    for isoform_ in isoform_list_:
        #--------------------
        X_train = np.load("./data/{}/train_data.npy".format(isoform_))
        X_val   = np.load("./data/{}/val_data.npy".format(isoform_))
        X_test  = np.load("./data/{}/test_data.npy".format(isoform_))

        y_train = np.load("./data/{}/train_label.npy".format(isoform_))
        y_val   = np.load("./data/{}/val_label.npy".format(isoform_))
        y_test  = np.load("./data/{}/test_label.npy".format(isoform_))
        #--------------------
        # Set up Parameter
        my_n_neighbors = np.arange(3,20,2)
        pred_val_list = []
        para_list = []
        for p in my_n_neighbors:
            para_list.append(p)
            my_classifier = make_pipeline(VarianceThreshold(threshold), 
                                          KNeighborsClassifier(n_neighbors=p))
            pred_val = my_classifier.fit(X_train, y_train).predict_proba(X_val)[::,1]
            pred_val_list.append(list(pred_val))
        #--------------------
        auc_val_list = []
        for pred in pred_val_list:
            auc = roc_auc_score(y_val, pred)
            auc_val_list.append(auc)
        i = np.argmax(auc_val_list)
        #--------------------
        best_n_neighbors = para_list[i]
        tuned_classifier = make_pipeline(VarianceThreshold(threshold), 
                                         KNeighborsClassifier(n_neighbors=best_n_neighbors))
        pred_test = tuned_classifier.fit(X_train, y_train).predict_proba(X_test)[::,1]
        #--------------------
        metric = printPerformance(y_test, pred_test)
        df1 = pd.DataFrame(metric)
        df = pd.concat([df, df1], axis=1)
    df.columns = ["Metrics"] + isoform_list_
    df.to_csv("knn_benchmark.csv", index=None)
    end = time.time()
    processing_time = (end - start)
    print("Processing time: {}".format(processing_time))
    
#==========================================================================================#                 
#                                 BENCHMARKING SVM MODEL                                   #           
#==========================================================================================#
def svm_benchmark(isoform_list):
    start = time.time()
    columns_= ['ACC', 'BA', 'ROC-AUC', 'PR-AUC', 'MCC', 'SN', 'SP', 'PR', 'F1', 'CK']
    df = pd.DataFrame(columns_)
    isoform_list_ = isoform_list
    for isoform_ in isoform_list_:
        #--------------------
        X_train = np.load("./data/{}/train_data.npy".format(isoform_))
        X_val   = np.load("./data/{}/val_data.npy".format(isoform_))
        X_test  = np.load("./data/{}/test_data.npy".format(isoform_))

        y_train = np.load("./data/{}/train_label.npy".format(isoform_))
        y_val   = np.load("./data/{}/val_label.npy".format(isoform_))
        y_test  = np.load("./data/{}/test_label.npy".format(isoform_))
        #--------------------
        # Set Up Parameter
        my_C     = [0.001, 0.01, 0.1, 1, 10, 100]
        my_gamma = [0.001, 0.01, 0.1, 1, 10, 100]
        pred_val_list = []
        para_list = []
        for p1 in my_C:
            for p2 in my_gamma:
                para_list.append((p1, p2))
                my_classifier = make_pipeline(VarianceThreshold(threshold), 
                                              SVC(C=p1, gamma=p2, probability=True))
                pred_val = my_classifier.fit(X_train, y_train).predict_proba(X_val)[::,1]
                pred_val_list.append(list(pred_val))
        #--------------------
        auc_val_list = []
        for pred in pred_val_list:
            auc = roc_auc_score(y_val, pred)
            auc_val_list.append(auc)
        i = np.argmax(auc_val_list)
        #--------------------
        best_C = para_list[i][0]
        best_gamma = para_list[i][1]
        tuned_classifier = make_pipeline(VarianceThreshold(threshold), 
                                         SVC(C=best_C, gamma=best_gamma, probability=True))
        pred_test = tuned_classifier.fit(X_train, y_train).predict_proba(X_test)[::,1]
        #--------------------
        metric = printPerformance(y_test, pred_test)
        df1 = pd.DataFrame(metric)
        df = pd.concat([df, df1], axis=1)
    df.columns = ["Metrics"] + isoform_list_
    df.to_csv("svm_benchmark.csv", index=None)
    end = time.time()
    processing_time = (end - start)
    print("Processing time: {}".format(processing_time))

#==========================================================================================#                 
#                                 BENCHMARKING RF MODEL                                    #           
#==========================================================================================#
def rf_benchmark(isoform_list):
    start = time.time()
    columns_= ['ACC', 'BA', 'ROC-AUC', 'PR-AUC', 'MCC', 'SN', 'SP', 'PR', 'F1', 'CK']
    df = pd.DataFrame(columns_)
    isoform_list_ = isoform_list
    for isoform_ in isoform_list_:
        #--------------------
        X_train = np.load("./data/{}/train_data.npy".format(isoform_))
        X_val   = np.load("./data/{}/val_data.npy".format(isoform_))
        X_test  = np.load("./data/{}/test_data.npy".format(isoform_))

        y_train = np.load("./data/{}/train_label.npy".format(isoform_))
        y_val   = np.load("./data/{}/val_label.npy".format(isoform_))
        y_test  = np.load("./data/{}/test_label.npy".format(isoform_))
        #--------------------
        # Set Up Parameter
        my_n_estimators = np.arange(25,201,25)
        pred_val_list = []
        para_list = []
        for p in my_n_estimators:
            para_list.append(p)
            my_classifier = make_pipeline(VarianceThreshold(threshold), 
                                          RandomForestClassifier(random_state=42, n_estimators=p))
            pred_val = my_classifier.fit(X_train, y_train).predict_proba(X_val)[::,1]
            pred_val_list.append(list(pred_val))
        #--------------------
        auc_val_list = []
        for pred in pred_val_list:
            auc = roc_auc_score(y_val, pred)
            auc_val_list.append(auc)
        i = np.argmax(auc_val_list)
        #--------------------
        best_n_estimators = para_list[i]
        tuned_classifier = make_pipeline(VarianceThreshold(threshold), 
                                         RandomForestClassifier(random_state=42, n_estimators=best_n_estimators))
        pred_test = tuned_classifier.fit(X_train, y_train).predict_proba(X_test)[::,1]
        #--------------------
        metric = printPerformance(y_test, pred_test)
        df1 = pd.DataFrame(metric)
        df = pd.concat([df, df1], axis=1)
    df.columns = ["Metrics"] + isoform_list_
    df.to_csv("rf_benchmark.csv", index=None)
    end = time.time()
    processing_time = (end - start)
    print("Processing time: {}".format(processing_time))