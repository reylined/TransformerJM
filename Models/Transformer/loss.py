import torch
import torch.nn as nn
import numpy as np


def long_loss(yhat, y, mask):
    mask = mask.repeat((1,1,y.shape[2]))
    loss_func = nn.MSELoss(reduction='none')
    #loss_func = nn.L1Loss(reduction='none')
    #loss_func = nn.SmoothL1Loss(reduction='none', beta=0.5)
    loss = loss_func(yhat, y)
    loss = loss * mask
    loss = loss.sum()/mask.sum()
    
    return loss



def surv_loss(surv_pred, mask, event):
    mask = mask.numpy()
    event = event.numpy()
    mask_out = mask[:,1:]
    mask_rev = mask_out[:,::-1]
    event_time_index = mask_out.shape[1] - np.argmax(mask_rev, axis=1) - 1
    
    e_filter = np.zeros([mask_out.shape[0],mask_out.shape[1]])
    for row_index, row in enumerate(e_filter):
        if event[row_index]:
            row[event_time_index[row_index]] = 1
    s_filter = mask_out - e_filter

    s_filter = torch.tensor(s_filter)
    e_filter = torch.tensor(e_filter)
    
    surv_pred = surv_pred.squeeze()
    nll_loss = torch.log(surv_pred)*s_filter + torch.log(1-surv_pred)*e_filter
    nll_loss = nll_loss.sum() / mask_out.sum()
    
    #nll_loss = nll_loss.sum(dim=1) / torch.tensor(mask_out.sum(axis=1))
    #nll_loss = nll_loss.mean()
    
    return -nll_loss
    