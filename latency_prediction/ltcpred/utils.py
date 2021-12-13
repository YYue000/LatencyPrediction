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

#https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)

def evaluate(results, error_percentage):
    c= np.zeros(len(error_percentage))
    error_percentage = np.array(error_percentage).reshape(-1,1)
    n = 0
    for res in results:
        latency = res['latency']
        prediction = res['prediction']
        delta = np.abs(latency-prediction)/latency
        #delta = np.abs(latency-prediction)/prediction
        c += np.sum(delta<=error_percentage, axis=1)
        n += len(latency)
    c = c/n
    return c

