import yaml
import copy
import numpy as np
import torch.optim as optim

def parse_config(cfg_name):
    cfg = None
    with open(cfg_name, 'r') as stream:
        cfg = yaml.safe_load(stream)
    return cfg

from models.gcn import GCN #, MLP

def build_model(cfg_model):
    model_type = cfg_model['type']
    if model_type == 'GCN':
        model = GCN(**cfg_model['kwargs'])
    elif model_type == 'mlp':
        model = MLP(**cfg_model['kwargs'])
    else:
        raise NotImplementedError
    
    return model


def build_optimizer(cfg_optim, model):
    cfg_optim = copy.deepcopy(cfg_optim)
    opt_func = cfg_optim.get('type', 'AdamW')
    cfg_optim['kwargs']['params'] = model.parameters()
    optimizer = getattr(optim, opt_func)(**cfg_optim['kwargs'])
    return optimizer

def build_lr_scheduler(cfg_lr, optimizer):
    cfg_lr = copy.deepcopy(cfg_lr)
    cfg_lr['kwargs']['optimizer'] = optimizer
    lr_scheduler = getattr(optim.lr_scheduler, cfg_lr['type'])(**cfg_lr['kwargs'])
    return lr_scheduler

def evaluate(results, error_percentage):
    c= np.zeros(len(error_percentage))
    error_percentage = np.array(error_percentage).reshape(-1,1)
    n = 0
    for res in results:
        latency = res['latency']
        prediction = res['prediction']
        delta = np.abs(latency-prediction)/latency
        c += np.sum(delta<error_percentage, axis=1)
        n += len(latency)
    c = c/n
    return c
