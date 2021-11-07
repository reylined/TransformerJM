import torch
import numpy as np
import pandas as pd

from sksurv.functions import StepFunction


def get_numpy(df, long = ["Y1","Y2","Y3"], base = ["X1","X2"], obstime = "obstime", max_len=None):
    df.loc[:,"id_new"] = df.groupby(by="id").grouper.group_info[0] # assign id from 0 to num subjects
    df.loc[:,"roundtime"] = (df.loc[:,obstime] * 2).round() / 2 # round obstime to nearest 0.5
    if "visit" not in df:
        df.loc[:,"visit"] = df.groupby(by="id").cumcount()
    
    I = len(np.unique(df.loc[:,"id"]))
    if max_len is None:
        max_len = int(np.max(df.loc[:,"roundtime"]) * 2 + 1) # based on 0.5 rounding

    x_long = np.empty((I, max_len, len(long)))
    x_long[:,:,:] = np.NaN
    x_base = np.zeros((I, len(base)))
    for index, row in df.iterrows():
        ii = int(row.loc["id_new"])
        jj = int(row.loc["roundtime"]) * 2 # based on 0.5 rounding
        x_long[ii,jj,:] = row.loc[long]
        if jj==0:
            x_base[ii,:] = row.loc[base]
    
    e = df.loc[df["visit"]==0,"event"].values.squeeze()
    t = df.loc[df["visit"]==0,"time"].values.squeeze()
    obs_time = np.arange(0, max_len/2, 0.5)
    
    return x_long, x_base, e, t, obs_time

def sortByTime(x, e, t):
    # sort data by descending survival time
    t_index_desc = torch.argsort(t, descending=True)
    x = x[t_index_desc]
    e = e[t_index_desc]
    t = t[t_index_desc]
    return x, e, t
    
    
## Loss function
def surv_loss(risk, event):
    hr = torch.exp(risk)
    log_risk = torch.log(torch.cumsum(hr,0))
    uncensored_LL = risk - log_risk
    censored_LL = uncensored_LL * event.unsqueeze(1)
    neg_likelihood = -torch.sum(censored_LL)# / torch.sum(event)
    return neg_likelihood


def _compute_counts(event, time, order=None):
    """Count right censored and uncensored samples at each unique time point.
    Parameters
    ----------
    event : array
        Boolean event indicator.
    time : array
        Survival time or time of censoring.
    order : array or None
        Indices to order time in ascending order.
        If None, order will be computed.
    Returns
    -------
    times : array
        Unique time points.
    n_events : array
        Number of events at each time point.
    n_at_risk : array
        Number of samples that have not been censored or have not had an event at each time point.
    n_censored : array
        Number of censored samples at each time point.
    """
    n_samples = event.shape[0]

    if order is None:
        order = np.argsort(time, kind="mergesort")

    uniq_times = np.empty(n_samples, dtype=time.dtype)
    uniq_events = np.empty(n_samples, dtype=np.int_)
    uniq_counts = np.empty(n_samples, dtype=np.int_)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    times = np.resize(uniq_times, j)
    n_events = np.resize(uniq_events, j)
    total_count = np.resize(uniq_counts, j)
    n_censored = total_count - n_events

    # offset cumulative sum by one
    total_count = np.r_[0, total_count]
    n_at_risk = n_samples - np.cumsum(total_count)

    return times, n_events, n_at_risk[:-1], n_censored


class BreslowEstimator:
    """Breslow's estimator of the cumulative hazard function.
    Attributes
    ----------
    cum_baseline_hazard_ : :class:`sksurv.functions.StepFunction`
        Cumulative baseline hazard function.
    baseline_survival_ : :class:`sksurv.functions.StepFunction`
        Baseline survival function.
    """

    def fit(self, linear_predictor, event, time):
        """Compute baseline cumulative hazard function.
        Parameters
        ----------
        linear_predictor : array-like, shape = (n_samples,)
            Linear predictor of risk: `X @ coef`.
        event : array-like, shape = (n_samples,)
            Contains binary event indicators.
        time : array-like, shape = (n_samples,)
            Contains event/censoring times.
        Returns
        -------
        self
        """
        risk_score = np.exp(linear_predictor)
        order = np.argsort(time, kind="mergesort")
        risk_score = risk_score[order]
        uniq_times, n_events, n_at_risk, _ = _compute_counts(event, time, order)

        divisor = np.empty(n_at_risk.shape, dtype=np.float_)
        value = np.sum(risk_score)
        divisor[0] = value
        k = 0
        for i in range(1, len(n_at_risk)):
            d = n_at_risk[i - 1] - n_at_risk[i]
            value -= risk_score[k:(k + d)].sum()
            k += d
            divisor[i] = value

        assert k == n_at_risk[0] - n_at_risk[-1]

        y = np.cumsum(n_events / divisor)
        self.cum_baseline_hazard_ = StepFunction(uniq_times, y)
        self.baseline_survival_ = StepFunction(self.cum_baseline_hazard_.x,
                                               np.exp(- self.cum_baseline_hazard_.y))
        return self

    def get_cumulative_hazard_function(self, linear_predictor):
        """Predict cumulative hazard function.
        Parameters
        ----------
        linear_predictor : array-like, shape = (n_samples,)
            Linear predictor of risk: `X @ coef`.
        Returns
        -------
        cum_hazard : ndarray, shape = (n_samples,)
            Predicted cumulative hazard functions.
        """
        risk_score = np.exp(linear_predictor)
        n_samples = risk_score.shape[0]
        funcs = np.empty(n_samples, dtype=np.object_)
        for i in range(n_samples):
            funcs[i] = StepFunction(x=self.cum_baseline_hazard_.x,
                                    y=self.cum_baseline_hazard_.y,
                                    a=risk_score[i])
        return funcs

    def get_survival_function(self, linear_predictor):
        """Predict survival function.
        Parameters
        ----------
        linear_predictor : array-like, shape = (n_samples,)
            Linear predictor of risk: `X @ coef`.
        Returns
        -------
        survival : ndarray, shape = (n_samples,)
            Predicted survival functions.
        """
        risk_score = np.exp(linear_predictor)
        n_samples = risk_score.shape[0]
        funcs = np.empty(n_samples, dtype=np.object_)
        for i in range(n_samples):
            funcs[i] = StepFunction(x=self.baseline_survival_.x,
                                    y=np.power(self.baseline_survival_.y, risk_score[i]))
        return funcs
