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
    """
    Computing and storing the average and current value
    """
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

def training(model, 
             train_loader, 
             valid_loader, 
             criterion, 
             optimizer, 
             scheduler, 
             num_training_steps: int = 1000, 
             loss_weights: List[float] = [0.6, 0.4],
             log_interval: int = 1,
             eval_internal: int = 1,
             savedir: str = None,
             use_wandb: bool = False,
             device: str = 'cpu') -> dict:
             
    batch_time_m   = AverageMeter()
    data_time_m    = AverageMeter()
    losses_m       = AverageMeter()
    l1_losses_m    = AverageMeter()
    focal_losses_m = AverageMeter()

    # Criterion
    l1_criterion, focal_criterion = criterion
    l1_weights, focal_weights     = loss_weights

    # Set training model
    model.train()

    # Set optimizer
    optimizer.zero_grad()

    # Set training
    best_score = 0
    step = 0
    train_mode = True

    while train_mode:
        end = time.time()
        for inputs, masks, targets in train_loader:
            # Send to device 'cpu' device or 'gpu' device
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)

            data_time_m.update(time.time() - end)

            # Predict
            outputs    = model(inputs)
            outputs    = F.softmax(outputs, dim=1)
            l1_loss    = l1_criterion(outputs[:, 1, :], masks)
            focal_loss = focal_criterion(outputs, masks)
            loss       = (l1_weights * l1_loss) + (focal_weights * focal_loss)

            loss.backward()

            # Update weights
            optimizer.step()
            optimizer.zero_grad()

            # Log loss
            l1_losses_m.update(l1_loss.item())
            focal_losses_m.update(focal_loss.item())
            losses_m.update(loss.item())

            batch_time_m.update(time.time() - end)

            # Wandb
            if use_wandb:
                wandb.log({
                    'lr': optimizer.param_group[0]['lr'],
                    'train_l1_loss': l1_losses_m.val,
                    'train_focal_loss': focal_losses_m.val,
                    'train_loss': losses_m.val}, 
                    step=step)

            if (step + 1) % log_interval == 0 or step == 0:
                _logger.info('TRAIN [{:>4d/{}]'
                             'Loss: {loss.val:>6.4f} ({loss.avg:>6.4f})'
                             'Focal Loss: {focal_loss.val:>6.4f} ({focal_loss.avg:>6.4f})'
                             'LR: {lr:.3e}'
                             'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)'
                             'Data: {data_time.val:.3f} ({data_time.avg:.3f}'.format(
                                step+1, num_training_steps, 
                                loss=losses_m, 
                                l1_loss=l1_losses_m, 
                                focal_loss=focal_losses_m,
                                lr=optimizer.param_groups[0]['lr'],
                                batch_time=batch_time_m,
                                rate=inputs.size(0) / batch_time_m.val,
                                rate_avg=inputs.size(0) / batch_time_m.avg,
                                data_time=data_time_m))

            if ((step+1) % eval_internal == 0 and step != 0) or (step+1) == num_training_steps:
                eval_metrics = evaluate(model, valid_loader, criterion, log_interval, device)
                model.train()

                eval_log = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])

                # Wandb
                if use_wandb:
                    wandb.log(eval_log, step=step)

                # Checkpoint
                if best_score < np.mean(list(eval_metrics.values())):
                    # Saving best score
                    state = {'best_step':step}
                    state.update(eval_log)
                    json.dump(state, open(os.path.join(savedir, 'best_score.json'), 'w'), indent='\t')

                    # Saving best model
                    torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))

                    _logger.info('Best Score {0:.3%} to {1:.3%}'.format(best_score, np.mean(list(eval_metrics.values()))))

                    best_score = np.mean(list(eval_metrics.values()))

            # Scheduler
            if scheduler is not None:
                scheduler.step()

            end = time.time()

            step += 1

            if step == num_training_steps:
                train_mode = False
                break
        
    # Print best score and step
    _logger.info('Best Metric: {0:.3%} (step {1:})',format(best_score, state['best_step']))

    # Save the latest model
    torch.save(model.state_dict(), os.path.join(savedir, f'latest_model.pt'))

    # Save the latest score
    state = {'latest_step': step}
    state.update(eval_log)
    json.dump(state, open(os.path.join(savedir, 'latest_score.json'), 'w'), indent='\t')


def evaluate(model, data_loader, criterion, log_interval, device='cpu'):
    auroc_image_metric = AUROC(num_classes=1, pos_label=1)
    auroc_pixel_metric = AUROC(num_classes=1, pos_label=1)
    aupro_pixel_metric = AUPRO()
    
    model.eval()

    with torch.no_grad():
        for idx, (inputs, masks, targets) in enumerate(data_loader):
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)

            # Predict
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            anomaly_score = torch.topk(torch.flatten(outputs[:, 1, :], start_dim=1), 100)[0].mean(dim=1)

            # Update metrics
            auroc_image_metric.update(
                preds=anomaly_score.cpu(),
                target=targets.cpu()
            )
            auroc_pixel_metric.update(
                preds=outputs[:,1,:].cpu(),
                target=masks.cpu()
            )
            aupro_pixel_metric.update(
                preds=outputs[:,1,:].cpu(),
                target=masks.cpu()
            )
    
    # Metrics
    metrics = {
        'AUROC-image': auroc_image_metric.compute().item(),
        'AUROC-pixel': auroc_pixel_metric.compute().item(),
        'AUPRO-pixel': aupro_pixel_metric.compute().item()
    }

    _logger.info('TEST: AUROC-image: %.3f%% | AUROC-pixel: %.3f%% | AUPRO-pixel: %.3f%%' %
                    (metrics['AUROC-image'], metrics['AUROC-pixel'], metrics['AUPRO-pixel']))

    return metrics