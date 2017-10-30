import scipy as sp
from sklearn.metrics import roc_auc_score

"""
    Evaluation functions, including precision, logloss & auc_score.
    Both parameters, act & pred, are numpy.ndarray types.
"""


def precision(act, pred):
    c = (act == pred)
    return float(len(c[c == True])) / len(c)


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(pred, epsilon)
    pred = sp.minimum(pred, 1-epsilon)
    ll = sum(act*sp.log(pred) + sp.subtract(1, act)*sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll


def auc_score(act, pred):
    auc = roc_auc_score(act, pred, average=None, sample_weight=None)
    return auc
