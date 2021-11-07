import torch

def positional_encoding(batch_size, length, d_model, obs_time):
    """
    Positional Encoding for each visit
    
    Parameters
    ----------
    batch_size:
        Number of subjects in batch
    length:
        Number of visits
    d_model:
        Dimension of the model vector
    obs_time:
        Observed/recorded time of each visit
    """
    PE = torch.zeros((batch_size, length, d_model))
    if obs_time.ndim == 0:
        obs_time = obs_time.repeat(batch_size).unsqueeze(1)
    elif obs_time.ndim == 1:
        obs_time = obs_time.repeat(batch_size,1)

    pow0 = torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32)/d_model)
    PE[:, :, 0::2] = torch.sin(torch.einsum('ij,k->ijk', obs_time, pow0))
    pow1 = torch.pow(10000, torch.arange(1, d_model, 2, dtype=torch.float32)/d_model)
    PE[:, :, 1::2] = torch.cos(torch.einsum('ij,k->ijk', obs_time, pow1))

    return PE



