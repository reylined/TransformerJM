import torch
import torch.nn as nn
import torch.nn.functional as F

class MATCH(nn.Module):
    def __init__(self, n_long, n_base, out_len):
        super().__init__()
        self.long1 = conv_block(n_long, 32, kernel_size=3, padding=1)
        self.mask1 = conv_block(n_long, 8, kernel_size=3, padding=1)
        
        self.long2 = conv_block(40, 32, kernel_size=3, padding=1)
        self.mask2 = conv_block(8, 8, kernel_size=3, padding=1)
        
        self.long3 = conv_block(40, 32, kernel_size=3, padding=1)
        self.mask3 = conv_block(8, 8, kernel_size=3, padding=1)
        
        self.long4 = conv_block(40, 32, kernel_size=3)
        
        self.survival = nn.Sequential(
            nn.Linear(32 + n_base, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64,out_len)
        )
            
    
    def forward(self, long, base, mask):
        x = self.long1(long)
        mask = self.mask1(mask)
        x = torch.cat((x,mask),dim=1)
        
        x = self.long2(x)
        mask = self.mask2(mask)
        x = torch.cat((x,mask),dim=1)
        
        x = self.long3(x)
        mask = self.mask3(mask)
        x = torch.cat((x,mask),dim=1)
        
        x = self.long4(x)
        
        x = F.adaptive_avg_pool1d(x,1).squeeze()
        x = torch.cat((x,base),dim=1)
        x = self.survival(x)
        return x
        

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, padding_mode='replicate', **kwargs),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.BatchNorm1d(out_channels)
            )
    def forward(self, x):
            return self.convolution(x)