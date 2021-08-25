# Import libraries
import numpy as np
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, cohen_kappa_score, balanced_accuracy_score

#===========================================================================================
# Get probabilities 
def get_prob(prob_list, best_epoch):
    bestE_problist = []
    for batch_y in prob_list[best_epoch]:
        for i in batch_y:
            bestE_problist.append(i.detach().cpu().numpy())
    bestE_problist = np.array(bestE_problist)
    return bestE_problist

#===========================================================================================
# Get model performance
def printPerformance(labels, probs, thresold=0.5, printout=False):
    #--------------------
    if thresold != 0.5:
        predicted_labels = []
        for prob in probs:
            if prob >= thresold:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
    else:
        predicted_labels = np.round(probs)
    #--------------------
    tn, fp, fn, tp = confusion_matrix(labels, predicted_labels).ravel()
    acc            = accuracy_score(labels, predicted_labels)
    ba             = balanced_accuracy_score(labels, predicted_labels)
    roc_auc        = roc_auc_score(labels, probs)
    pr_auc         = average_precision_score(labels, probs)
    mcc            = matthews_corrcoef(labels, predicted_labels)
    sensitivity    = tp / (tp + fn)
    specificity    = tn / (tn + fp)
    precision      = tp / (tp + fp)
    f1             = 2*precision*sensitivity / (precision + sensitivity)
    ck             = cohen_kappa_score(labels, predicted_labels)
    if printout:
        print('Accuracy: ', round(acc, 4))
        print('AUC-ROC: ', round(roc_auc, 4))
        print('AUC-PR: ', round(pr_auc, 4))
        print('MCC: ', round(mcc, 4))
        print('Sensitivity/Recall: ', round(sensitivity, 4))
        print('Specificity: ', round(specificity, 4))
        print('Precision: ', round(precision, 4))
        print('F1-score: ', round(f1, 4))
    return round(acc, 4), round(ba, 4), round(roc_auc, 4), round(pr_auc, 4), round(mcc, 4), round(sensitivity, 4), round(specificity, 4), round(precision, 4), round(f1, 4), round(ck, 4) 