import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .predictor import Predictor
from .initializer import initialize

class GraphConvolution(nn.Module):
    def __init__(self,  feature_shape, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.Tensor(feature_shape))
        if bias:
            self.bias = Parameter(torch.Tensor(feature_shape[1]))
        else:
            self.register_parameter('bias', None)

    def forward(self, adj_matrix, x):
        out = torch.bmm(adj_matrix, torch.matmul( x, self.weight))
        if self.bias is not None:
            out = out + self.bias
        return out

def gc_norm_relu_drop(feature_shape, dropout_rate):
    return nn.Sequential(
        GraphConvolution(feature_shape),
        nn.LayerNorm(feature_shape[1]),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate))
        
class GCN(Predictor):
    def __init__(self, depth, 
                    feature_dim, hidden_dim, augments_dim, dropout_rate,
                    initializer=None,
                    criterion_cfg=None):
        super(GCN, self).__init__(criterion_cfg)

        self.gcs = nn.Sequential(
            *[gc_norm_relu_drop([feature_dim if _ == 0 else hidden_dim, hidden_dim], dropout_rate) 
                for _ in range(depth)])
        self.fc = nn.Linear(hidden_dim+augments_dim, 1)
        
        if initializer is not None:
            initializer_gc = initializer['gc']
            initialize(self.gcs, GraphConvolution, **initializer_gc)
            initializer_fc = initializer.get('fc', None)
            if initializer_fc is not None:
                initialize(self.fc, nn.Linear,  **initializer_fc)
        
    def forward(self, input, return_loss=None):
        adjency = input['adjacency']
        x = input['features']
        augments = input.get('augments', None)
        y = self._forward(adjency, x, augments)
        if not self.training and not return_loss:
            return y, None
            
        target = input['latency']
        loss = self.compute_loss(y, target)
        if self.training:
            return loss
        else:
            return y, loss
    
    def _forward(self, adjency, x, augments=None):
        x = self.gcs(adjency, x)
        x = x[:,0] # use global node
        if augments is not None:
            x = torch.cat([x, augments], dim=1)
            
        y = self.fc(x)
        return y
    
    
