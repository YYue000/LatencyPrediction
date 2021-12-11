import argparse
import copy
import logging
import torch
from torch.optim import lr_scheduler, optimizer
from dataset import build_dataloader

from utils import parse_config, build_model, build_optimizer, build_lr_scheduler, evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s#%(lineno)d:%(message)s')
logger = logging.getLogger('global')


def train(model, cfg):
    train_data = build_dataloader(cfg['data'], 'train')
    val_data = build_dataloader(cfg['data'], 'val')
    
    epochs = cfg['trainer']['epochs']
    optimizer = build_optimizer(cfg['trainer']['optimizer'], model)
    step_on_val_loss_epoch = cfg['trainer']['optimizer'].get('step_on_val_loss_epoch', epochs+1)
    lr_scheduler = build_lr_scheduler(cfg['trainer']['lr_scheduler'], optimizer)
    optimizer.zero_grad()

    for epoch in range(epochs):
        model.train()
        for it, input in enumerate(train_data):
            loss = model(input)
            loss.backward()
            optimizer.step()
        
        # validation
        _, accs, val_loss = test(model, val_data, cfg, return_loss=True)
        if epoch > step_on_val_loss_epoch:
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()
        logger.info(f'epoch: {epoch} val acc: {accs}')
        
    model.train() # model is at training at the end of train()
    return model


def test(model, test_data, cfg, return_loss=False):
    # inference
    model.eval()
    results = []
    all_loss = 0 if return_loss else None
    N = len(test_data)
    with torch.no_grad():
        for it, input in enumerate(test_data):
            y, loss = model.forward(input, return_loss)
            if loss is not None:
                all_loss += loss/N
            output = {input[k] for k in ['arch_id', 'arch']}
            output['prediction'] = y.numpy()
            latency = input.get('latency', None)
            if latency is not None:
                output['latency'] = latency.numpy()
            results.append(output)
        
    # evaluation
    accs = evaluate(results, cfg['leeways'])
    return results, accs, all_loss

def get_cfg(cfg_raw, mode):
    cfg_runtime = copy.deepcopy(cfg_raw)
    cfg_runtime.update(cfg_runtime[mode])
    return cfg_runtime

def main(args):
    cfg_raw = parse_config(args.cfg_file)
    cfg_test = get_cfg(cfg_raw, 'test')
    if not args.test:
        cfg = get_cfg(cfg_raw, 'train')
    else:
        cfg = cfg_test
    
    model = build_model(cfg['model'])
    
    if not args.test:
        train(model, cfg)
        cfg = cfg_test

    test_data = build_dataloader(cfg['data'], 'test')
    test(model, test_data, cfg)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='An implementation of latency prediction')
    parser.add_argument('--cfg', dest='cfg_file', required=True)
    parser.add_argument('-t', dest='test', action='store_true')
    args = parser.parse_args()
    print(args)
    main(args)
