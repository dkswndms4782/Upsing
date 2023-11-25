from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import torch
import torch.nn.functional as F 


def get_metric(targets, preds):
    ## 오호 roc_auc_score는 확률값 그대로 넣고, acc는 진짜값과 예측값을 넣어서 계산하는듯
    auc_preds = F.softmax(torch.from_numpy(preds), dim=1)
    auc = roc_auc_score(targets, auc_preds, multi_class='ovr')
    acc = accuracy_score(targets, torch.argmax(torch.from_numpy(preds), 1))

    return auc, acc