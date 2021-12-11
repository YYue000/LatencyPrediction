from models.gcn import GCN

def build_model(cfg_model):
    model_type = cfg_model['type']
    if model_type == 'GCN':
        model = GCN(**cfg_model['kwargs'])
    elif model_type == 'mlp':
        model = MLP(**cfg_model['kwargs'])
    else:
        raise NotImplementedError
    
    return model