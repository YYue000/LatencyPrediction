import torch.nn as nn

def initialize_normal(m, std):
    nn.init.normal_(m.weight.data, std=std)
    if m.bias is not None:
        m.bias.data.zero_()

def initialize_xavier_normal(m):
    nn.init.xavier_normal_(m.weight.data)
    if m.bias is not None:
        m.bias.data.zero_()

def initialize_xavier_uniform(m):
    nn.init.xavier_uniform_(m.weight.data)
    if m.bias is not None:
        m.bias.data.zero_()

def initialize_orthogonal(m):
    nn.init.orthogonal_(m)
    if m.bias is not None:
        m.bias.data.zero_()

def initialize(module, ModuleType, method, **kwargs):
    for m in module.modules():
        if isinstance(m, ModuleType):
            if method == 'normal':
                initialize_normal(m, **kwargs)
            elif method == 'xavier_normal':
                initialize_xavier_normal(m, **kwargs)
            elif method == 'xavier_uniform':
                initialize_xavier_uniform(m, **kwargs)
            elif method == 'orthogonal':
                initialize_orthogonal(m, **kwargs)
            else:
                raise NotImplementedError
