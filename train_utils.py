import logging
import os, json
import time, wandb
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from anomalib.utils.metrics import AUPRO, AUROC

_logger = logging.getLogger('train')

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def training(model, trainloader, validloader, 
             criterion, optimizer, scheduler, 
             num_training_steps: int = 1000, 
             loss_weights: List[float] = [0.6, 0.4], 
             log_interval: int = 1, 
             eval_interval: int = 1, 
             savedir: str = None, use_wandb: bool = False, 
             device: str ='cpu') -> dict:

    batch_time_m   = AverageMeter()
    data_time_m    = AverageMeter()
    losses_m       = AverageMeter()
    l1_losses_m    = AverageMeter()
    focal_losses_m = AverageMeter()
    
    # Criterion
    l1_criterion, focal_criterion = criterion
    l1_weight, focal_weight       = loss_weights

    best_score = 0
    global_step = 0
    
    # Begin training
    for _step in range(num_training_steps):
        model.train()
        end = time.time()
        for inputs, masks, targets in trainloader:
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)

            data_time_m.update(time.time() - end)

            # set optimizer
            optimizer.zero_grad()
            
            # Get prediction
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            l1_loss = l1_criterion(outputs[:,1,:], masks)
            focal_loss = focal_criterion(outputs, masks)
            loss = (l1_weight * l1_loss) + (focal_weight * focal_loss)

            loss.backward()

            optimizer.step()

            # Log loss
            l1_losses_m.update(l1_loss.item())
            focal_losses_m.update(focal_loss.item())
            losses_m.update(loss.item())
            batch_time_m.update(time.time() - end)

            # wandb
            if use_wandb:
                wandb.log({
                    'lr':optimizer.param_groups[0]['lr'],
                    'train_focal_loss': focal_losses_m.val,
                    'train_l1_loss': l1_losses_m.val,
                    'train_loss': losses_m.val
                },
                step=_step)

            if (_step+1) % log_interval == 0 or _step == 0: 
                _logger.info('TRAIN [{:>4d}/{}] '
                            'Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                            'L1 Loss: {l1_loss.val:>6.4f} ({l1_loss.avg:>6.4f}) '
                            'Focal Loss: {focal_loss.val:>6.4f} ({focal_loss.avg:>6.4f}) '
                            'LR: {lr:.3e} '
                            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            _step+1, num_training_steps, 
                            loss       = losses_m, 
                            l1_loss    = l1_losses_m,
                            focal_loss = focal_losses_m,
                            lr         = optimizer.param_groups[0]['lr'],
                            batch_time = batch_time_m,
                            rate       = inputs.size(0) / batch_time_m.val,
                            rate_avg   = inputs.size(0) / batch_time_m.avg,
                            data_time  = data_time_m))

        # Evaluation
        if ((_step+1) % eval_interval == 0 and _step != 0) or (_step+1) == num_training_steps:
            # Eval on validation set
            eval_metrics = evaluate(model, validloader, criterion, log_interval, device)
            val_score = np.mean(list(eval_metrics.values()))
            
            eval_log = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])

            # wandb
            if use_wandb:
                wandb.log(eval_log, step=_step)

            # checkpoint
            if best_score < val_score:
                # save best score
                state = {'best_step':_step}
                state.update(eval_log)
                json.dump(state, open(os.path.join(savedir, 'best_score.json'),'w'), indent='\t')

                # save best model
                torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
                
                _logger.info('Best Score {0:.3%} to {1:.3%}'.format(best_score, val_score))

                best_score = val_score
                
            # save latest model
            torch.save(model.state_dict(), os.path.join(savedir, f'latest_model.pt'))
            # save latest score
            state = {'latest_step':_step}
            state.update(eval_log)
            json.dump(state, open(os.path.join(savedir, 'latest_score.json'),'w'), indent='\t')

        scheduler.step()

    # print best score and step
    _logger.info('Best Metric: {0:.3%} (step {1:})'.format(best_score, state['best_step']))



    
def evaluate(model, dataloader, criterion, log_interval, device='cpu'):
    model.eval()
    # Evaluation metrics
    auroc_image_metric = AUROC(num_classes=1, pos_label=1)
    auroc_pixel_metric = AUROC(num_classes=1, pos_label=1)
    aupro_pixel_metric = AUPRO()
    with torch.no_grad():
        for idx, (inputs, masks, targets) in enumerate(dataloader):
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            anomaly_score = torch.topk(torch.flatten(outputs[:,1,:], start_dim=1), 100)[0].mean(dim=1)

            # update metrics
            auroc_image_metric.update(
                preds  = anomaly_score.cpu(), 
                target = targets.cpu()
            )
            auroc_pixel_metric.update(
                preds  = outputs[:,1,:].cpu(),
                target = masks.cpu()
            )
            aupro_pixel_metric.update(
                preds   = outputs[:,1,:].cpu(),
                target  = masks.cpu()
            )
    # metrics    
    metrics = {
        'AUROC-image':auroc_image_metric.compute().item(),
        'AUROC-pixel':auroc_pixel_metric.compute().item(),
        'AUPRO-pixel':aupro_pixel_metric.compute().item()

    }
    _logger.info("\n================================")
    _logger.info('TEST: AUROC-image: %.3f%% | AUROC-pixel: %.3f%% | AUPRO-pixel: %.3f%%' % 
                (metrics['AUROC-image'] * 100, metrics['AUROC-pixel'] * 100, metrics['AUPRO-pixel'] * 100))
    _logger.info("\n================================")

    return metrics