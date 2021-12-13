import argparse
import os
import copy
import logging
import torch
from dataset import build_dataloader

from utils import parse_config, build_model, build_optimizer, build_lr_scheduler, evaluate
from utils import EarlyStopping

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s#%(lineno)d:%(message)s')
logger = logging.getLogger('global')


def train(model, cfg):
    train_data = build_dataloader(cfg['data'], 'train')
    val_data = build_dataloader(cfg['data'], 'val')
    
    epochs = cfg['trainer']['epochs']
    st_epoch = 0
    best_val_acc = 0
    best_epoch = None

    optimizer = build_optimizer(cfg['trainer']['optimizer'], model)
    step_on_val_loss = (cfg['trainer']['lr_scheduler']['type'] in ['ReduceLROnPlateau']) 
    step_on_val_loss_epoch = cfg['trainer']['lr_scheduler'].get('step_on_val_loss_epoch', -1)
    lr_scheduler = build_lr_scheduler(cfg['trainer']['lr_scheduler'], optimizer)

    es = EarlyStopping(**cfg['trainer']['early_stopping']['kwargs'])
    es_start_epoch = cfg['trainer']['early_stopping']['start_epoch']

    resume = cfg['trainer'].get('resume', None)
    if resume is not None:
        ckpt = torch.load(resume)
        logger.info(f'resuming from {resume}')
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        st_epoch = ckpt['epoch'] + 1
    
    save_freq = cfg['trainer'].get('save_freq', 1)
    
    for epoch in range(st_epoch, epochs):
        model.train()
        for input in train_data:
            loss = model(input)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # validation
        _, accs, val_loss = test(model, val_data, cfg, return_loss=True)
        if step_on_val_loss:
            if epoch > step_on_val_loss_epoch:
                lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()
        logger.info(f'epoch: {epoch}/{epochs} train loss: {loss:.4f} val loss: {val_loss:.4f} val acc: {accs}')
        if accs[0] > best_val_acc:
            best_epoch = epoch
            best_val_acc = accs[0]
            torch.save({'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'val_accs': accs,
                        'optimizer': optimizer.state_dict(),
                        'lr_sceduler': lr_scheduler.state_dict(),
                        'cfg': cfg
                        },
                    f'checkpoints/ckpt_best.pth')

        if epoch % save_freq == 0 or epoch == epochs-1:
            torch.save({'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'val_accs': accs,
                        'optimizer': optimizer.state_dict(),
                        'lr_sceduler': lr_scheduler.state_dict(),
                        'cfg': cfg
                        },
                    f'checkpoints/ckpt_epoch{epoch}.pth')


        if epoch > es_start_epoch:
            if es.step(val_loss):
                logger.info('Early stopping criterion is met, stop training now.')
                break
        
    model.train() # model is at training at the end of train()
    logger.info(f'best epoch {best_epoch} best_acc {best_val_acc}')
    return model


def test(model, test_data, cfg, return_loss=False):
    # inference
    model.eval()

    resume = cfg['trainer'].get('resume', None)
    if resume is not None:
        ckpt = torch.load(resume)
        logger.info(f'loading ckpt from {resume}')
        model.load_state_dict(ckpt['state_dict'])

    results = []
    all_loss = 0 if return_loss else None
    N = len(test_data)
    with torch.no_grad():
        for input in test_data:
            y, loss = model.forward(input, return_loss)
            if loss is not None:
                all_loss += loss/N
            output = {k:input[k] for k in ['arch_id', 'arch']}
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
        save_path = 'checkpoints/ckpt.pth'
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        train(model, cfg)

        cfg = cfg_test
    else:
        assert cfg['trainer'].get('resume', None) is not None

    test_data = build_dataloader(cfg['data'], 'test')

    results, accs, __ = test(model, test_data, cfg)
    logger.info(f'{accs}')

    if args.dump:
        import pickle
        output_file = 'results/results.pickle'
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        output_results = [{'arch':r['arch'][_],
                            'prediction':r['prediction'][_],
                            'latency':r['latency'][_]} 
                        for r in results
                        for _ in range(len(r['arch']))] 
        pickle.dump(output_results, open(output_file, 'wb'))
        with open(output_file.replace('pickle', 'txt'), 'w') as fw:
            for r in output_results:
                fw.write(f'{r}\n')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='An implementation of latency prediction')
    parser.add_argument('--cfg', dest='cfg_file', required=True)
    parser.add_argument('-t', dest='test', action='store_true')
    parser.add_argument('--dump', dest='dump', action='store_true')
    args = parser.parse_args()
    logger.info(args)
    main(args)
