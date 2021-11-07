import torch
import numpy as np


def get_tensors(df, long=["Y1","Y2","Y3"], base=["X1","X2"], obstime = "obstime"):
    '''
    Changes batch data from dataframe to corresponding tensors for Transformer model

    Parameters
    ----------
    df : Pandas Dataframe

    Returns
    -------
    x :
        3d tensor of data with shape (I (subjects), J (visits w/ padding), K (covariates))
    e,t :
        1d tensor of event indicator and event times (I)
    mask :
        2d tensor (1-obs, 0-padding) with shape (I, J)
    obs_time:
        2d tensor of observation times with shape (I, J)

    '''
    df.loc[:,"id_new"] = df.groupby(by="id").grouper.group_info[0]
    if "visit" not in df:
        df.loc[:,"visit"] = df.groupby(by="id").cumcount()
    
    I = len(np.unique(df.loc[:,"id"]))
    max_len = np.max(df.loc[:,"visit"]) + 1
    
    x_base = torch.zeros(I, max_len, len(base))
    x_long = torch.zeros(I, max_len, len(long))
    mask = torch.zeros((I, max_len), dtype=torch.bool)
    obs_time = torch.zeros(I, max_len)
    for index, row in df.iterrows():
        ii = int(row.loc["id_new"])
        jj = int(row.loc["visit"])

        x_base[ii,jj,:] = torch.tensor(row.loc[base])
        x_long[ii,jj,:] = torch.tensor(row.loc[long])
        mask[ii,jj] = 1
        obs_time[ii,jj] = row.loc[obstime]
   
    e = torch.tensor(df.loc[df["visit"]==0,"event"].values).squeeze()
    t = torch.tensor(df.loc[df["visit"]==0,"time"].values).squeeze()
    
    return x_long, x_base, mask, e, t, obs_time



def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)



def get_mask(pad = None, future = True, window = None):
    size = pad.shape[-1]
    mask = (pad != 0).unsqueeze(-2)
    if future:
        future_mask = np.triu(np.ones((1,size,size)), k=1).astype('uint8')==0
        if window is not None:
            win_mask = np.triu(np.ones((1,size,size)), k=-window+1).astype('uint8')==1
            future_mask = future_mask & win_mask
        mask = mask & future_mask
    return mask



class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, model_size, warmup, factor):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(optimizer, d_model, warmup_steps=200, factor=1):
    return NoamOpt(optimizer, d_model, warmup_steps, factor)
  