# shall code the metrics
# Precision, Recall, F1-score, ROC, AUC, mAP, mAR, confusion matrix
from collections import defaultdict
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import random
import time
import numpy as np
def accuracy(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    return (true==pred).sum()/true.size

def precision(true, pred):
    conf_matrix = defaultdict(int)
    for t,p in zip(true, pred):
        if t==1 and p == 1:
            conf_matrix["tp"] += 1
        elif t == 1 and p ==0:
            conf_matrix["fn"] += 1
        elif t == 0 and p == 1:
            conf_matrix["fp"] += 1
        elif t == 0 and p ==0:
            conf_matrix["tn"] += 1
    # return conf_matrix
    precision = conf_matrix["tp"]/ (conf_matrix["tp"] + conf_matrix["fp"])
    recall = conf_matrix["tp"]/ (conf_matrix["tp"] + conf_matrix["fn"])
    return precision, recall

            

length = 10000
true = [random.randint(0,1) for _ in range(length)]
predicted = [random.randint(0,1) for _ in range(length)]
print("metrics:")
start = time.time()
print(precision(true, predicted))
print(time.time()-start, "seconds")
print("=="*20)
print("sklearn precision_recall_score:")
start = time.time()
print(precision_score(true, predicted), recall_score(true, predicted))
print(time.time()-start, "seconds")
# confusion-matrix:
