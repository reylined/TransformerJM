# TransformerJM

Implementation and simulation study for TransformerJM and other neural network based models for multivariate longitudinal and survival data.

See `file_description.txt` for a description of each code file.

## Example

First, we import the necessary libraries.

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

The global options are set first. For the simulation, we set the number of simulated subjects to be `I` and the times at which longitudinal observations are made to be `obstime`. Setting `scenario = "none"` equates to Scenario 1 described in the paper. During evaluation, predictions are made using the longitudinal observations up to each `landmark_time` for times `pred_windows` into the future.

```python
# Global options
n_sim = 1
I = 1000
obstime = [0,1,2,3,4,5,6,7,8,9,10]
landmark_times = [1,2,3,4,5]
pred_windows = [1,2,3]
scenario = "none"
```



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

train_long, train_base, train_mask, e_train, t_train, train_obs_time = get_tensors(train_data.copy())
```



```python
## Train model
torch.manual_seed(0)

model = Transformer(d_long=3, d_base=2, d_model=32, nhead=4,
                    num_decoder_layers=7)
model.apply(init_weights)
model = model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = get_std_opt(optimizer, d_model=32, warmup_steps=200, factor=0.2)

n_epoch = 25
batch_size = 32

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
