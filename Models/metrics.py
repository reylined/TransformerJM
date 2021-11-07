import torch
import numpy as np
import warnings

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
ro.r.source("PATH/AUC_BS.r")
AUC_R = ro.globalenv['AUC']
Brier_R = ro.globalenv['Brier']


def get_integrated(x, times):
    return np.trapz(x,times) / (max(times)-min(times))

def AUC(x, event, time, pred_times):
    auc = AUC_R(surv=x, event=event, time=time, predtimes=pred_times)[0]
    iauc = get_integrated(auc, pred_times)
    return auc, iauc

def Brier(x, event, time, event_train, time_train, LT, pred_windows):
    bs = Brier_R(surv=x, event=event, time=time,
                 event_train=event_train, time_train=time_train,
                 LT = LT, DeltaT=pred_windows)
    ibs = get_integrated(bs, pred_windows)
    return bs, ibs


def MSE(y, yhat):
    mse = np.square(y-yhat)
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        mse = np.nanmean(mse, axis=1) # average over time
    mse = np.nanmean(mse, axis=0) # average over subj
    return mse