import torch
import torch.optim as optim

import sys
sys.path.append("PATH")
from Models.MFPCA_DeepSurv.DeepSurv import DeepSurv
from Models.MFPCA_DeepSurv.functions import (get_numpy, sortByTime, surv_loss, BreslowEstimator)
from Models.metrics import (AUC, Brier)
from Simulation.data_simulation_base import simulate_JM_base
from Simulation.data_simulation_nonPH import simulate_JM_nonPH

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
ro.r.source("PATH/MFPCA.r")
mfpca_train = ro.globalenv['mfpca_train']
mfpca_test = ro.globalenv['mfpca_test']

import pickle
import time
start = time.time()


# Global options
n_sim = 100
I = 1000
obstime = [0,1,2,3,4,5,6,7,8,9,10]
landmark_times = [1,2,3,4,5]
pred_windows = [1,2,3]


# Initialize arrays for storing results
AUC_array = np.zeros((n_sim, len(landmark_times), len(pred_windows)))
iAUC_array = np.zeros((n_sim, len(landmark_times)))
true_AUC_array = np.zeros((n_sim, len(landmark_times), len(pred_windows)))
true_iAUC_array = np.zeros((n_sim, len(landmark_times)))

BS_array = np.zeros((n_sim, len(landmark_times), len(pred_windows)))
iBS_array = np.zeros((n_sim, len(landmark_times)))
true_BS_array = np.zeros((n_sim, len(landmark_times), len(pred_windows)))
true_iBS_array = np.zeros((n_sim, len(landmark_times)))



for i_sim in range(n_sim):
    if i_sim % 10 == 0:
        print("i_sim:",i_sim)
    
    np.random.seed(i_sim)
    data_all = simulate_JM_base(I=I, obstime=obstime, opt="none", seed=i_sim)
    data = data_all[data_all.obstime < data_all.time]

    
    ## split train/test
    random_id = range(I) #np.random.permutation(range(I))
    train_id = random_id[0:int(0.7*I)]
    test_id = random_id[int(0.7*I):I]

    train_data = data[data["id"].isin(train_id)]
    test_data = data[data["id"].isin(test_id)]

    ## Scale data using Min-Max Scaler
    minmax_scaler = MinMaxScaler(feature_range=(-1,1))
    train_data.loc[:,["Y1","Y2","Y3"]] = minmax_scaler.fit_transform(train_data.loc[:,["Y1","Y2","Y3"]])
    test_data.loc[:,["Y1","Y2","Y3"]] = minmax_scaler.transform(test_data.loc[:,["Y1","Y2","Y3"]])
    
    x_long_train, x_base_train, e_train, t_train, obs_time = get_numpy(train_data)
    
    mfpca_out = mfpca_train(x_long_train, obs_time)
    mfpca_scores = np.array(mfpca_out[0])
    Cms = mfpca_out[1]
    psis = mfpca_out[2]
    x_train = np.concatenate((mfpca_scores, x_base_train), axis=1)
    x_train = torch.FloatTensor(x_train)
    e_train_t = torch.FloatTensor(e_train)
    t_train_t = torch.FloatTensor(t_train)

    y_train = list(zip(e_train.flatten().tolist(),t_train.flatten().tolist()))
    y_train = np.array(y_train,dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    
    ## Train model
    torch.manual_seed(0)
    model = DeepSurv(x_train.shape[1])
    model = model.train()

    optimizer = optim.Adam(model.parameters())
    
    n_epoch = 12
    batch_size = 32
    
    loss_values = []
    for epoch in range(n_epoch):
        running_loss = 0
        permutation = torch.randperm(x_long_train.shape[0])
        for batch in range(0, x_long_train.shape[0], batch_size):
            optimizer.zero_grad()
            
            indices = permutation[batch:batch+batch_size]
            batch_x, batch_e, batch_t = \
                x_train[indices,:], e_train_t[indices], t_train_t[indices]
            batch_x, batch_e, batch_t = sortByTime(
                batch_x, batch_e, batch_t)
            if len(indices)>1: #drop if last batch size is 1
                risk = model.forward(batch_x)
                loss = surv_loss(risk, batch_e)
                loss.backward()
                optimizer.step()
                running_loss += loss
        loss_values.append(running_loss.tolist())
    plt.plot(loss_values)
    
    S_func = BreslowEstimator().fit(model(x_train).detach().numpy(), e_train, t_train)
    
    
    ## Test model
    for LT_index, LT in enumerate(landmark_times):
        
        pred_times = [x+LT for x in pred_windows]
        
        # Only keep subjects with survival time > landmark time
        tmp_data = test_data.loc[test_data["time"]>LT,:]
        tmp_id = np.unique(tmp_data["id"].values)
        tmp_all = data_all.loc[data_all["id"].isin(tmp_id),:]
        
        # Only keep longitudinal observations <= landmark time
        tmp_data = tmp_data.loc[tmp_data["obstime"]<=LT,:]
        
        true_prob_tmp = tmp_all.loc[tmp_all["predtime"].isin(pred_times), ["true"]].values.reshape(-1,len(pred_times))
        true_prob_LT = tmp_all.loc[tmp_all["predtime"]==LT, ["true"]].values
        true_prob_tmp = true_prob_tmp / true_prob_LT
        
        x_long_tmp, x_base_tmp, e_tmp, t_tmp, obs_time = get_numpy(tmp_data, max_len=len(obs_time))
        
        y_tmp = list(zip(e_tmp.flatten().tolist(),t_tmp.flatten().tolist()))
        y_tmp = np.array(y_tmp,dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
        
        # MFPCA on testing data
        mfpca_out_test = mfpca_test(x_long_train, x_long_tmp, obs_time, Cms, psis)
        mfpca_scores_test = np.array(mfpca_out_test[0])
        
        x_test = np.concatenate((mfpca_scores_test, x_base_tmp), axis=1)
        x_test = torch.FloatTensor(x_test)

        # Survival prediction        
        model = model.eval()
        risk = model(x_test)
        risk = risk.view(-1).detach().numpy()
        S_hat = S_func.get_survival_function(risk)
        surv_pred = np.array([si(pred_times) for si in S_hat])

        
        auc, iauc = AUC(surv_pred, e_tmp, t_tmp, np.array(pred_times))
        AUC_array[i_sim, LT_index, :] = auc
        iAUC_array[i_sim, LT_index] = iauc
        auc, iauc = AUC(true_prob_tmp, e_tmp, t_tmp, np.array(pred_times))
        true_AUC_array[i_sim, LT_index, :] = auc
        true_iAUC_array[i_sim, LT_index] = iauc
        
        bs, ibs = Brier(surv_pred, e_tmp, t_tmp,
                          e_train, t_train, LT, np.array(pred_windows))
        BS_array[i_sim, LT_index, :] = bs
        iBS_array[i_sim, LT_index] = ibs
        bs, ibs = Brier(true_prob_tmp, e_tmp, t_tmp,
                          e_train, t_train, LT, np.array(pred_windows))
        true_BS_array[i_sim, LT_index, :] = bs
        true_iBS_array[i_sim, LT_index] = ibs



np.set_printoptions(precision=3)
print("AUC:\n",np.nanmean(AUC_array, axis=0))
print("iAUC:",np.nanmean(iAUC_array, axis=0))
print("True AUC:\n",np.nanmean(true_AUC_array, axis=0))
print("True iAUC:",np.nanmean(true_iAUC_array, axis=0))
print("Difference:",np.mean(true_iAUC_array, axis=0) - np.mean(iAUC_array, axis=0))

print("BS:\n", np.mean(BS_array, axis=0))
print("iBS:",np.mean(iBS_array, axis=0))
print("True BS:\n", np.mean(true_BS_array, axis=0))
print("True iBS:",np.mean(true_iBS_array, axis=0))


end = time.time()
print("total time:", (end-start)/60)



'''
## save results
results = {"AUC":AUC_array,
           "iAUC":iAUC_array,
           "True_AUC":true_AUC_array,
           "True_iAUC":true_iAUC_array,
           "BS":BS_array,
           "iBS":iBS_array,
           "True_BS":true_BS_array,
           "True_iBS":true_iBS_array}

outfile = open('MFPCA_DS.pickle', 'wb')
pickle.dump(results, outfile)
outfile.close() 
'''

'''
## read results
infile = open('MFPCA_DS.pickle', 'rb')
results = pickle.load(infile)
infile.close
'''
