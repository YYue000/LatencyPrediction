import torch.nn as nn
import torch.nn.functional as F
from functools import partial

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
        else:
            raise NotImplementedError

        self.loss_weight = criterion_cfg.get('loss_weight', 1.0)
    
    def forward(self, input):
        raise NotImplementedError
    
    def compute_loss(self, y, target):
        return self.criterion(y,target)*self.loss_weight