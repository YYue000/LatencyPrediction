import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .predictor import Predictor
from .initializer import initialize

import logging
logger = logging.getLogger('global')

class GraphConvolution(nn.Module):
    def __init__(self,  in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.DoubleTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.DoubleTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, adj_matrix, x):
        ## for debug
        t=torch.matmul(x, self.weight)
        logger.info(f'{t.shape} {adj_matrix.shape}')


        out = torch.bmm(adj_matrix, torch.matmul(x, self.weight))
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return  f'{self.__class__.__name__} (in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})'

class GCNormReLUDrop(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super(GCNormReLUDrop, self).__init__()
        self.gc = GraphConvolution(in_features, out_features)
        self.ln = nn.LayerNorm(out_features).double()
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, adjacency, x):
        x = self.gc(adjacency, x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.drop(x)
        return x


class GCN(Predictor):
    def __init__(self, depth, 
                    feature_dim, hidden_dim, augments_dim, dropout_rate,
                    initializer=None,
                    criterion_cfg=None):
        super(GCN, self).__init__(criterion_cfg)

        self.gcs = nn.ModuleList(
            [GCNormReLUDrop(feature_dim if _ == 0 else hidden_dim, hidden_dim, dropout_rate) 
                for _ in range(depth)])
        self.fc = nn.Linear(hidden_dim+augments_dim, 1)
        
        if initializer is not None:
            initializer_gc = initializer['gc']
            initialize(self.gcs, GraphConvolution, **initializer_gc)
            initializer_fc = initializer.get('fc', None)
            if initializer_fc is not None:
                initialize(self.fc, nn.Linear,  **initializer_fc)
        
        logger.info(f'model {self}')

    def forward(self, input, return_loss=None):
        logger.info(f'{input["arch"]}')
        adjacency = input['adjacency']
        x = input['features']
        augments = input.get('augments', None)
        y = self._forward(adjacency, x, augments)
        if not self.training and not return_loss:
            return y, None
            
        target = input['latency']
        loss = self.compute_loss(y, target)
        if self.training:
            return loss
        else:
            return y, loss
    
    def _forward(self, adjacency, x, augments=None):
        for gc in self.gcs:
            x = gc(adjacency, x)
        x = x[:,0] # use global node
        if augments is not None:
            x = torch.cat([x, augments], dim=1)
            
        y = self.fc(x)
        return y
    
    
