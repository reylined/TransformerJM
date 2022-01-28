# TransformerJM

Implementation and simulation study for TransformerJM and other neural network based models for multivariate longitudinal and survival data.

See `file_description.txt` for a description of each code file.

## Annotated Example

The following is a walkthrough of the TransformerJM method on a simulated dataset. We demonstrate how the model is trained and used for prediction. First, we import the necessary libraries.

```python
# Pytorch
import torch
import torch.nn as nn

# Source TransformerJM code
import sys
sys.path.append("your_path")
from Models.Transformer.TransformerJM import Transformer
from Models.Transformer.functions import (get_tensors, get_mask, init_weights, get_std_opt)
from Models.Transformer.loss import (long_loss, surv_loss)
from Models.metrics import (AUC, Brier, MSE)
from Simulation.data_simulation_base import simulate_JM_base

# Other Python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None
```

The global options are set. For the simulation, we set the number of simulated subjects to be `I` and the times at which longitudinal observations are made to be `obstime`. Setting `scenario = "none"` equates to Scenario 1 described in the paper. During evaluation, predictions are made using the longitudinal observations up to each `landmark_time` for times `pred_windows` into the future.

```python
# Global options
n_sim = 1
I = 1000
obstime = [0,1,2,3,4,5,6,7,8,9,10]
landmark_times = [1,2,3,4,5]
pred_windows = [1,2,3]
scenario = "none" # ["none", "interaction", "nonph"]
```

To simulate the data, call the `simulate_JM_base` function and keep only the observations which occur up to the time of event. In general, the data should be in the "long" format where each row contains the observations for a subject at a specific time. Here we split the data, using 70% for training and setting aside the remaining 30% for testing. The data is standardized using a min-max scaler.

```python
data_all = simulate_JM_base(I=I, obstime=obstime, opt=scenario, seed=i_sim)
data = data_all[data_all.obstime <= data_all.time]

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
```

The model is first instantiated from the Transformer class.
Before training, we define an optimizer which manages the learning rate during training. We use the [Noam Optimizer](https://stackoverflow.com/questions/65343377/adam-optimizer-with-warmup-on-pytorch) which is a common choice for Transformers.

```python
## Train model
torch.manual_seed(0)

model = Transformer(d_long=3, d_base=2, d_model=32, nhead=4,
                    num_decoder_layers=7)
model.apply(init_weights)
model = model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = get_std_opt(optimizer, d_model=32, warmup_steps=200, factor=0.2)
```

In Pytorch, models are trained in a training loop. An in depth tutorial can be found on the [Pytorch Website](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html).
Specific to TransformerJM is the use of the `get_tensors` helper function which takes the data in the form of a long dataframe and returns it in tensor format.
The model takes the data and returns predictions for the longitudinal outcomes and survival probabilities. These are given as input to the `long_loss` and `surv_loss` functions respectively. Backpropagation is computed with respect to the sum of these two loss functions.

```python
n_epoch = 25
batch_size = 32

loss_values = []
for epoch in range(n_epoch):
    running_loss = 0
    train_id = np.random.permutation(train_id)
    for batch in range(0, len(train_id), batch_size):
        optimizer.zero_grad()

        indices = train_id[batch:batch+batch_size]
        batch_data = train_data[train_data["id"].isin(indices)]

        batch_long, batch_base, batch_mask, batch_e, batch_t, obs_time = get_tensors(batch_data.copy())
        batch_long_inp = batch_long[:,:-1,:]
        batch_long_out = batch_long[:,1:,:]
        batch_base = batch_base[:,:-1,:]
        batch_mask_inp = get_mask(batch_mask[:,:-1])
        batch_mask_out = batch_mask[:,1:].unsqueeze(2)

        yhat_long, yhat_surv = model(batch_long_inp, batch_base, batch_mask_inp,
                     obs_time[:,:-1], obs_time[:,1:])
        loss1 = long_loss(yhat_long, batch_long_out, batch_mask_out)
        loss2 = surv_loss(yhat_surv, batch_mask, batch_e)
        loss = loss1 + loss2
        loss.backward()
        scheduler.step()
        running_loss += loss
    loss_values.append(running_loss.tolist())
plt.plot((loss_values-np.min(loss_values))/(np.max(loss_values)-np.min(loss_values)), 'b-')
```


```python
## Survival Prediction using Landmarking
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
    true_prob_tmp = true_prob_tmp / true_prob_LT # true conditional survival

    tmp_long, tmp_base, tmp_mask, e_tmp, t_tmp, obs_time = get_tensors(tmp_data.copy())

    base_0 = tmp_base[:,0,:].unsqueeze(1)        
    long_0 = tmp_long
    mask_T = torch.ones((long_0.shape[0],1), dtype=torch.bool)

    dec_long = long_0
    dec_base = base_0

    long_pred = torch.zeros(long_0.shape[0],0,long_0.shape[2])
    surv_pred = torch.zeros(long_0.shape[0],0,1)

    model = model.eval()

    for pt in pred_times:
        dec_base = base_0.expand([-1,dec_long.shape[1],-1])

        out = model.decoder(dec_long, dec_base, get_mask(tmp_mask), obs_time)
        out = model.decoder_pred(out[:,-1,:].unsqueeze(1), out, tmp_mask.unsqueeze(1), torch.tensor(pt))
        long_out = model.long(out)
        surv_out = torch.sigmoid(model.surv(out))

        long_pred = torch.cat((long_pred, long_out), dim=1)
        surv_pred = torch.cat((surv_pred, surv_out), dim=1)

        dec_long = torch.cat((dec_long, long_out), dim=1)
        tmp_mask = torch.cat((tmp_mask, mask_T), dim=1)
        obs_time = torch.cat((obs_time, torch.tensor(pt).expand([obs_time.shape[0],1])),dim=1)

    long_pred = long_pred.detach().numpy()
    surv_pred = surv_pred.squeeze().detach().numpy()
    surv_pred = surv_pred.cumprod(axis=1)

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
```
