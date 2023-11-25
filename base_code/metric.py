from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import torch
import torch.nn.functional as F 


def get_metric(targets, preds):
    ## ��ȣ roc_auc_score�� Ȯ���� �״�� �ְ�, acc�� ��¥���� �������� �־ ����ϴµ�
    auc_preds = F.softmax(torch.from_numpy(preds), dim=1)
    auc = roc_auc_score(targets, auc_preds, multi_class='ovr')
    acc = accuracy_score(targets, torch.argmax(torch.from_numpy(preds), 1))

    return auc, acc