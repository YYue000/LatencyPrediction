import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

def MAPELoss(output, target):
    return torch.sum(torch.abs((target - output) / target))

def RPDLoss(output, target):
    return torch.sum(torch.abs(target - output) / (torch.abs(target +output) / 2))

def MAPELoss_Diff(output, target):
    output = output[1:] - output[:-1]
    target = target[1:] - target[:-1]
    return torch.mean(torch.abs((target - output) / target))

def RPDLoss_Diff(output, target):
    output = output[1:] - output[:-1]
    target = target[1:] - target[:-1]
    return torch.mean(torch.abs(target - output) / ((torch.abs(target) + torch.abs(output)) / 2))


class Predictor(nn.Module):
    def __init__(self, criterion_cfg=None):
        super(Predictor, self).__init__()
        
        self.criterion = None
        self.loss_weight = None
        self._set_criterion(criterion_cfg)
    
    def _set_criterion(self, criterion_cfg):
        if criterion_cfg['type'] == 'L1':
            self.criterion = nn.L1Loss(reduction='sum')
        elif criterion_cfg['type'] == 'L2':
            self.criterion = nn.MSELoss(reduction='sum')
        elif criterion_cfg['type'] == 'SmoothL1':
            self.criterion = nn.SmoothL1Loss(reduction='sum')
        elif criterion_cfg['type'] == 'MAPE':
            self.criterion = MAPELoss
        elif criterion_cfg['type'] == 'RPD':
            self.criterion = RPDLoss
        else:
            raise NotImplementedError

        self.loss_weight = criterion_cfg.get('loss_weight', 1.0)
    
    def forward(self, input):
        raise NotImplementedError
    
    def compute_loss(self, y, target):
        B = y.shape[0]
        return self.criterion(y,target)*self.loss_weight/B
