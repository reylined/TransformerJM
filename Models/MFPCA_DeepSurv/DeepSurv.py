import torch.nn as nn

class DeepSurv(nn.Module):
    def __init__(self, n_features, hidden_size=64):
        super().__init__()        
        self.survival = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size,1)
        )
        
    def forward(self, features):
        x = self.survival(features)
        return x
        

