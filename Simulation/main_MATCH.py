import torch
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

import sys
sys.path.append("PATH")
from Models.MATCH.MATCH import MATCH
from Models.MATCH.functions import (get_tensors, augment, format_output, CE_loss)
from Models.metrics import (AUC, Brier)
from Simulation.data_simulation_base import simulate_JM_base
from Simulation.data_simulation_nonPH import simulate_JM_nonPH

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None

import pickle
import time
start = time.time()


n_sim = 2
I = 1000
obstime = [0,1,2,3,4,5,6,7,8,9,10]

landmark_times = [1,2,3,4,5]
pred_windows = [1,2,3]

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
    train_data.loc[:,["X1","X2","Y1","Y2","Y3"]] = minmax_scaler.fit_transform(train_data.loc[:,["X1","X2","Y1","Y2","Y3"]])
    test_data.loc[:,["X1","X2","Y1","Y2","Y3"]] = minmax_scaler.transform(test_data.loc[:,["X1","X2","Y1","Y2","Y3"]])
    
    train_long, train_base, train_mask, e_train, t_train, train_obs_time = get_tensors(train_data.copy()) # for BS


    ## Train model
    torch.manual_seed(0)

    out_len = 4
    model = MATCH(3,2, out_len)
    model = model.train()
    optimizer = optim.Adam(model.parameters())

    n_epoch = 25
    batch_size = 32
    
    test_long, test_base, test_mask, e_test, t_test, test_obs_time = get_tensors(test_data.copy())
    test_long, test_base, test_mask, e_test, t_test, subjid_test = augment(
        test_long, test_base, test_mask, e_test, t_test)
    
    loss_values = []
    loss_test = []
    for epoch in range(n_epoch):
        running_loss = 0
        train_id = np.random.permutation(train_id)
        for batch in range(0, len(train_id), batch_size):
            optimizer.zero_grad()
            
            indices = train_id[batch:batch+batch_size]
            batch_data = train_data[train_data["id"].isin(indices)]
            
            batch_long, batch_base, batch_mask, batch_e, batch_t, obs_time = get_tensors(batch_data.copy())
            batch_long, batch_base, batch_mask, batch_e, batch_t, subjid = augment(
                batch_long, batch_base, batch_mask, batch_e, batch_t)
            
            if len(indices)>1: #drop if last batch size is 1
                yhat_surv = torch.softmax(model(batch_long, batch_base, batch_mask), dim=1)
                s_filter, e_filter = format_output(obs_time, batch_mask, batch_t, batch_e, out_len)
                loss = CE_loss(yhat_surv, s_filter, e_filter)
                loss.backward()
                optimizer.step()
                running_loss += loss
        yhat_surv_test = torch.softmax(model(test_long, test_base, test_mask), dim=1)
        s_filter_t, e_filter_t = format_output(test_obs_time, test_mask, t_test, e_test, out_len)
        loss_t = CE_loss(yhat_surv_test, s_filter_t, e_filter_t)
        loss_test.append(loss_t.tolist())
        loss_values.append(running_loss.tolist())
    plt.plot((loss_values-np.min(loss_values))/(np.max(loss_values)-np.min(loss_values)), 'b-')
    plt.plot((loss_test-np.min(loss_test))/(np.max(loss_test)-np.min(loss_test)), 'g-')
    

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
        tmp_long, tmp_base, tmp_mask, e_tmp, t_tmp, obs_time = get_tensors(tmp_data.copy())
        
        model = model.eval()
        surv_pred = torch.softmax(model(tmp_long, tmp_base, tmp_mask), dim=1)
        surv_pred = surv_pred.detach().numpy()
        surv_pred = surv_pred[:,::-1].cumsum(axis=1)[:,::-1]
        surv_pred = surv_pred[:,1:(out_len+1)]
        
        auc, iauc = AUC(surv_pred, e_tmp.numpy(), t_tmp.numpy(), np.array(pred_times))
        AUC_array[i_sim, LT_index, :] = auc
        iAUC_array[i_sim, LT_index] = iauc
        auc, iauc = AUC(true_prob_tmp, np.array(e_tmp), np.array(t_tmp), np.array(pred_times))
        true_AUC_array[i_sim, LT_index, :] = auc
        true_iAUC_array[i_sim, LT_index] = iauc

        bs, ibs = Brier(surv_pred, e_tmp.numpy(), t_tmp.numpy(),
                          e_train.numpy(), t_train.numpy(), LT, np.array(pred_windows))
        BS_array[i_sim, LT_index, :] = bs
        iBS_array[i_sim, LT_index] = ibs
        bs, ibs = Brier(true_prob_tmp, e_tmp.numpy(), t_tmp.numpy(),
                          e_train.numpy(), t_train.numpy(), LT, np.array(pred_windows))
        true_BS_array[i_sim, LT_index, :] = bs
        true_iBS_array[i_sim, LT_index] = ibs


np.set_printoptions(precision=3)
print("AUC:",np.nanmean(AUC_array, axis=0))
print("iAUC:",np.mean(iAUC_array, axis=0))
print("True AUC:",np.nanmean(true_AUC_array, axis=0))
print("True iAUC:",np.mean(true_iAUC_array, axis=0))

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

outfile = open('MATCH_results.pickle', 'wb')
pickle.dump(results, outfile)
outfile.close() 
'''

'''
## read results
infile = open('MATCH_results.pickle', 'rb')
results = pickle.load(infile)
infile.close
'''





