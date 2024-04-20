# shall code the metrics
# Precision, Recall, F1-score, ROC, AUC, mAP, mAR, confusion matrix
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import random
import time
import numpy as np

def accuracy(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    if true.size:
        return (true==pred).sum()/true.size
    return 0


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

def precision_recall(true, pred, average="macro"):
    # generalising for multi class
    cls_count = Counter(true)
    true_set = set(true)
    true_set.update(set(pred))
    cls_unq = len(true_set)
    cm = np.zeros((cls_unq, cls_unq), dtype=np.int8)
    
    conf_matrix = defaultdict(int)
    for t,p in zip(true, pred):
        cm[t][p] += 1

    # class wise precision and recall

    if average=="micro":
        all_tps = []
        all_tps_fps = []
        # Micro Averaging
        for ids in true_set:
            all_tps.append(cm[ids][ids])
            all_tps_fps.append(cm[ids, :].sum())
        
        print("Micro Precision  | sklearn")
        print(np.sum(all_tps)/ np.sum(all_tps_fps), "|", precision_score(true, pred, average="micro"))
        print("Micro Recall     | sklearn")
        print(np.sum(all_tps)/ np.sum(all_tps_fps), "|", recall_score(true, pred, average="micro"))

    if average=="macro":
        precision_all = []
        recall_all = []
        # Macro Averaging
        for ids in true_set:
            precision_all.append(cm[ids][ids]/cm[:,ids].sum())
            recall_all.append(cm[ids][ids]/cm[ids,:].sum())
        print("Macro Precision | sklearn", )
        print(np.mean(precision_all,dtype = np.float64), "|", precision_score(true, pred, average="macro"))
        print("Macro Recall | sklearn", )
        print(np.mean(recall_all,dtype = np.float64), "|", recall_score(true, pred, average="macro"))


    if average == "weighted":
        precision_all = []
        recall_all = []
        # Weighted Averaging
        for ids in true_set:
            precision_all.append(cm[ids][ids]/cm[:,ids].sum() * (cls_count[ids]/len(true)))
            recall_all.append(cm[ids][ids]/cm[ids, :].sum() * (cls_count[ids]/len(true)))

        print("Weight Precision | sklearn", )
        print(np.sum(precision_all), "|", precision_score(true, pred, average="weighted"))
        print("Weight Recall | sklearn",)
        print(np.sum(recall_all), "|", recall_score(true, pred, average="weighted"))

    return ""

length = 100
true = [random.randint(0,2) for _ in range(length)]
predicted = [random.randint(0,2) for _ in range(length)]
print("metrics:")
start = time.time()
print("=="*20)
print(precision_recall(true, predicted, average="micro"))
print("=="*20)
print(precision_recall(true, predicted, average="macro"))
print("=="*20)
print(precision_recall(true, predicted, average="weighted"))
