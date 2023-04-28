"""
Calculate the Youden index NPV PPV ..., etc
"""

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
import pandas as pd
import numpy as np


def youden_index(y_true, y_score, pos_label=1, step=5):
    """
    clinical diagnostic indicators includes: PPV NPV Sensitivity Specific youden-index and TP FP TN FN
    YoudenIndex = sensitivity + specific - 1
    F1 = 2 * ppv * sentitivity / (ppv + sentitivity)
    input:
        y_true: the true binary labels for the classification, 1d numpy list
        y_score: the predicted scores or probabilities
        pos_label: 1 for positive class, and 0 for negative class
        step: the step of threshold should be >0 , if None will using the raw threshold returned by roc_curve
    return:
        df: DataFrame, corresponding indicators under different thresholds
        max_ji_val: the max youden value
        max_f1_val: the max F1
        roc_auc: auc under the roc area

    """
    fpr, tpr, thr = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    if step:
        thrs = list(map(lambda x: float(x) / 100, range(0, 100, int(step))))
    else:
        # Adding 0.99 to supplement the upper limit
        thr = [round(e, 2) for e in thr if e < 1] + [0.99]
        thrs = sorted(set(thr), reverse=False)

    # tn/fp/tp/fpr in diff threshold
    # print("阈值\tACC\tPPV\tNPV\t敏感度\t特异性\t约登指数\tF1\t真实良性数\t真实恶性数\t预测良性数\t预测恶性数\t真阳TP\t假阳FP\t真阴TN\t假阴FN")
    cols = "Thr\tACC\tPPV\tNPV\tSens(Rec/TPR)\tSpec\tYoudenIdx\tF1\tTrueBen\tTrueMal\tPredBen\tPredMal\tTP\tFP\tTN\tFN"
    # print(cols)
    columns = cols.split('\t')

    result = list()
    for i, f in enumerate(thrs):
        y_pred = np.zeros(y_score.shape[0], dtype=np.int8)
        idx = np.where(y_score >= f)  # notice: there is >=
        y_pred[idx] = 1
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        _fpr, _tpr = 0.0, 0.0
        if fp + tn != 0:
            _fpr = fp / (fp + tn)
        if tp + fn != 0:
            _tpr = tp / (tp + fn)  # sensitiveness
        ppv, npv = 0.0, 0.0
        if tp + fp != 0:
            ppv = tp / (tp + fp)
        if tn + fn != 0:
            npv = tn / (tn + fn)

        spec = 1.0 - _fpr  # specificity
        acc = (tp + tn) / float(tn + fp + fn + tp)
        jord_idx = _tpr + spec - 1  # youden index
        f1 = 2 * ppv * _tpr / (ppv + _tpr)
        s = "{:.2f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:2}\t{:2}\t{:2}\t{:2}\t{:2}\t{:2}\t{:2}\t{:2}"\
            .format(f, acc, ppv, npv, _tpr, spec, jord_idx, f1, tp + fn, fp + tn, tp + fp, tn + fn, tp, fp, tn, fn)
        # print(s)
        # 阈值	综合准确率	恶性准确率ppv	敏感度	良性准确率npv	特异性	约登指数	预测良性数	真实良性数	预测恶性数	真实恶性数
        row = [f, acc, ppv, npv, _tpr, spec, jord_idx, f1, tp + fn, fp + tn, tp + fp, tn + fn, tp, fp, tn, fn]
        result.append(row)

    df = pd.DataFrame(result, columns=columns)

    max_ji_val = df['YoudenIdx'].max()
    max_f1_val = df['F1'].max()

    return df, max_ji_val, max_f1_val, roc_auc

