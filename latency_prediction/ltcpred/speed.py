import argparse
import os
import copy
import logging
import torch
import time
from dataset import build_dataloader

from utils import parse_config, build_model, build_optimizer, build_lr_scheduler, evaluate
from utils import EarlyStopping

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s#%(lineno)d:%(message)s')
logger = logging.getLogger('global')

def test(model, test_data, cfg, return_loss=False):
    # inference
    model.eval()

    resume = cfg['trainer'].get('resume', None)
    if resume is not None:
        ckpt = torch.load(resume)
        model.load_state_dict(ckpt['state_dict'])

    results = []
    all_loss = 0 if return_loss else None
    N = len(test_data)
    with torch.no_grad():
        for input in test_data:
            y, loss = model.forward(input, return_loss)
       

def get_cfg(cfg_raw, mode):
    cfg_runtime = copy.deepcopy(cfg_raw)
    cfg_runtime.update(cfg_runtime[mode])
    return cfg_runtime

def main(args):
    cfg_raw = parse_config(args.cfg_file)
    cfg = get_cfg(cfg_raw, 'test')
    
    t0 = time.time()
    model = build_model(cfg['model'])

    test_data = build_dataloader(cfg['data'], 'test')

    test(model, test_data, cfg)
    t = time.time()
    logger.info(f'time {t-t0}')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='An implementation of latency prediction')
    parser.add_argument('--cfg', dest='cfg_file', required=True)
    args = parser.parse_args()
    logger.info(args)
    main(args)
