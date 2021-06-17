import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
from AFSD.common.anet_dataset import ANET_Dataset, detection_collate
from torch.utils.data import DataLoader
from AFSD.anet.BDNet import BDNet
from AFSD.anet.multisegment_loss import MultiSegmentLoss
from AFSD.common.config import config

from collections import OrderedDict
from bisect import bisect_right
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler

class WarmupMultiEpochLR(_LRScheduler):
    def __init__(self, optimizer, 
                    epoch_iters, 
                    warmstones, 
                    milestones, 
                    warmup_factor=0.01, 
                    gamma=0.1, 
                    warmup_method="constant",
                    last_epoch=-1
                ):
        
        self.milestones    = [item for item in milestones] # Counter(milestones)
        self.gamma         = gamma
        self.warmup_factor = warmup_factor
        self.warmstones    = warmstones 
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if(self.last_epoch < self.warmstones):
            if(self.warmup_method == "linear"):
                alpha = self.last_epoch / self.warmstones
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                warmup_factor = self.warmup_factor
                
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

batch_size      = config['training']['batch_size']
learning_rate   = config['training']['learning_rate']
weight_decay    = config['training']['weight_decay']
max_epoch       = config['training']['max_epoch']
checkpoint_path = config['training']['checkpoint_path']
focal_loss  = config['training']['focal_loss']
random_seed = config['training']['random_seed']
ngpu        = config['ngpu']

num_classes = config['dataset']['num_classes']

train_state_path = os.path.join(checkpoint_path, 'training')
if not os.path.exists(train_state_path):
    os.makedirs(train_state_path)

resume = config['training']['resume']
config['training']['ssl'] = 0.1

def print_training_info():
    print('batch size: ', batch_size)
    print('learning rate: ', learning_rate)
    print('weight decay: ', weight_decay)
    print('max epoch: ', max_epoch)
    print('checkpoint path: ', checkpoint_path)
    print('loc weight: ', config['training']['lw'])
    print('cls weight: ', config['training']['cw'])
    print('piou: ', config['training']['piou'])
    print('resume: ', resume)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

GLOBAL_SEED = 1

def worker_init_fn(worker_id):
    set_seed(GLOBAL_SEED + worker_id)

def get_rng_states():
    states = []
    states.append(random.getstate())
    states.append(np.random.get_state())
    states.append(torch.get_rng_state())
    if torch.cuda.is_available():
        states.append(torch.cuda.get_rng_state())
    return states

def set_rng_state(states):
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(states[3])

def save_model(epoch, model, optimizer):
    torch.save(model.module.state_dict(), 
                os.path.join(checkpoint_path, 'checkpoint-{}.ckpt'.format(epoch)))
    torch.save({'optimizer': optimizer.state_dict(), 'state': get_rng_states()}, 
                os.path.join(train_state_path, 'checkpoint_{}.ckpt'.format(epoch)))

def resume_training(resume, model, optimizer):
    start_epoch = 1
    if resume > 0:
        start_epoch += resume
        model_path = os.path.join(checkpoint_path, 'checkpoint-{}.ckpt'.format(resume))
        model.module.load_state_dict(torch.load(model_path))
        train_path = os.path.join(train_state_path, 'checkpoint_{}.ckpt'.format(resume))
        state_dict = torch.load(train_path)
        optimizer.load_state_dict(state_dict['optimizer'])
        set_rng_state(state_dict['state'])
    return start_epoch

def calc_bce_loss(start, end, scores):
    start = torch.tanh(start).mean(-1)
    end = torch.tanh(end).mean(-1)
    loss_start = F.binary_cross_entropy(start.view(-1), scores[:, 1].contiguous().view(-1).cuda(), reduction='mean')
    loss_end = F.binary_cross_entropy(end.view(-1), scores[:, 2].contiguous().view(-1).cuda(), reduction='mean')
    return loss_start, loss_end

def forward_one_epoch(net, clips, targets, scores=None, training=True, ssl=True):
    clips   = clips.cuda()
    targets = [t.cuda() for t in targets]

    if training:
        if ssl:
            output_dict = net(clips, proposals=targets, ssl=ssl)
        else:
            output_dict = net(clips, ssl=False)
    else:
        with torch.no_grad():
            output_dict = net(clips)

    if ssl:
        anchor, positive, negative = output_dict
        loss_ = []
        weights = [1, 0.1, 0.1]
        for i in range(3):
            loss_.append(nn.TripletMarginLoss()(anchor[i], positive[i], negative[i]) * weights[i])
        trip_loss = torch.stack(loss_).sum(0)
        return trip_loss
    else:
        loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct = CPD_Loss(
            [output_dict['loc'], output_dict['conf'],
             output_dict['prop_loc'], output_dict['prop_conf'],
             output_dict['center'], output_dict['priors']],
            targets)
        loss_start, loss_end = calc_bce_loss(output_dict['start'], output_dict['end'], scores)
        scores_ = F.interpolate(scores, scale_factor=1.0 / 8)
        loss_start_loc_prop, loss_end_loc_prop = calc_bce_loss(output_dict['start_loc_prop'],
                                                               output_dict['end_loc_prop'],
                                                               scores_)
        loss_start_conf_prop, loss_end_conf_prop = calc_bce_loss(output_dict['start_conf_prop'],
                                                                 output_dict['end_conf_prop'],
                                                                 scores_)
        loss_start = loss_start + 0.1 * (loss_start_loc_prop + loss_start_conf_prop)
        loss_end   = loss_end + 0.1 * (loss_end_loc_prop + loss_end_conf_prop)
        return loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct, loss_start, loss_end

def run_one_epoch(epoch, net, optimizer, data_loader, epoch_step_num, training=True):
    if training:
        net.train()
    else:
        net.eval()

    loss_loc_val     = 0
    loss_conf_val    = 0
    loss_prop_l_val  = 0
    loss_prop_c_val  = 0
    loss_ct_val      = 0
    loss_start_val   = 0
    loss_end_val     = 0
    loss_trip_val    = 0
    loss_contras_val = 0
    cost_val         = 0
    with tqdm.tqdm(data_loader, total=epoch_step_num, ncols=0) as pbar:
        for n_iter, (clips, targets, scores, ssl_clips, ssl_targets, flags) in enumerate(pbar):
            loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct, loss_start, loss_end = forward_one_epoch(
                net, clips, targets, scores, training=training, ssl=False)

            loss_l = loss_l * config['training']['lw']
            loss_c = loss_c * config['training']['cw']
            loss_prop_l = loss_prop_l * config['training']['lw']
            loss_prop_c = loss_prop_c * config['training']['cw']
            loss_ct = loss_ct * config['training']['cw']
            cost = loss_l + loss_c + loss_prop_l + loss_prop_c + loss_ct + loss_start + loss_end

            if flags[0]:
                loss_trip  = forward_one_epoch(net, ssl_clips, ssl_targets, training=training, ssl=True)
                loss_trip *= config['training']['ssl']
                cost = cost + loss_trip
                loss_trip_val += loss_trip.cpu().detach().numpy()

            if training:
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

            loss_loc_val    += loss_l.cpu().detach().numpy()
            loss_conf_val   += loss_c.cpu().detach().numpy()
            loss_prop_l_val += loss_prop_l.cpu().detach().numpy()
            loss_prop_c_val += loss_prop_c.cpu().detach().numpy()
            loss_ct_val     += loss_ct.cpu().detach().numpy()
            loss_start_val  += loss_start.cpu().detach().numpy()
            loss_end_val    += loss_end.cpu().detach().numpy()
            cost_val        += cost.cpu().detach().numpy()
            pbar.set_postfix(loss='{:.5f}'.format(float(cost.cpu().detach().numpy())))

    loss_loc_val    /= (n_iter + 1)
    loss_conf_val   /= (n_iter + 1)
    loss_prop_l_val /= (n_iter + 1)
    loss_prop_c_val /= (n_iter + 1)
    loss_ct_val     /= (n_iter + 1)
    loss_start_val  /= (n_iter + 1)
    loss_end_val    /= (n_iter + 1)
    loss_trip_val   /= (n_iter + 1)
    cost_val        /= (n_iter + 1)

    if training:
        prefix = 'Train'
        save_model(epoch, net, optimizer)
    else:
        prefix = 'Val'

    plog = 'Epoch-{} {} Loss: Total - {:.5f}, loc - {:.5f}, conf - {:.5f}, prop_loc - {:.5f}, ' \
           'prop_conf - {:.5f}, IoU - {:.5f}, start - {:.5f}, end - {:.5f}'.format(
        i, prefix, cost_val, loss_loc_val, loss_conf_val, loss_prop_l_val, loss_prop_c_val,
        loss_ct_val, loss_start_val, loss_end_val
    )
    plog = plog + ', Triplet - {:.5f}'.format(loss_trip_val)
    print(plog)

def load_param(model, file):
    checkpoint = torch.load(file, map_location='cpu')
    
    if('state_dict' in checkpoint.keys()):
        checkpoint = checkpoint['state_dict']
    
    model_state_dict = model.state_dict()
    new_state_dict   = OrderedDict()
    for k, v in checkpoint.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
        if name in model_state_dict:
            if v.shape != model_state_dict[name].shape:
                print('Skip loading parameter {}, required shape{}, '\
                      'loaded shape{}.'.format(name, model_state_dict[name].shape, v.shape))
                new_state_dict[name] = model_state_dict[name]
        else:
            print('Drop parameter {}.'.format(name))

    for key in model_state_dict.keys():
        if(key not in new_state_dict.keys()):
            print('No param {}.'.format(key))
            new_state_dict[key] = model_state_dict[key]
        
    model.load_state_dict(new_state_dict, strict=False)

if __name__ == '__main__':
    print_training_info()
    set_seed(random_seed)
    """
    Setup model
    """
    net = BDNet(in_channels=config['model']['in_channels'], backbone_model=config['model']['backbone_model'])
    net = nn.DataParallel(net, device_ids=list(range(ngpu))).cuda()
    if(config['model']['in_channels'] == 2):
        load_param(net.module, 'models/anet_flow/anet_flow_pre.ckpt') 
    else:
        load_param(net.module, 'models/anet/anet_rgb_pre.ckpt')
    
    """
    Setup loss
    """
    piou = config['training']['piou']
    CPD_Loss = MultiSegmentLoss(num_classes, piou, 1.0, use_focal_loss=focal_loss)

    """
    Setup dataloader
    """
    train_dataset = ANET_Dataset(config['dataset']['training']['video_info_path'],
                                 config['dataset']['training']['video_mp4_path'],
                                 config['dataset']['training']['clip_length'],
                                 config['dataset']['training']['crop_size'],
                                 config['dataset']['training']['clip_stride'],
                                 channels=config['model']['in_channels'],
                                 binary_class=False)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=8, worker_init_fn=worker_init_fn,
                                   collate_fn=detection_collate, pin_memory=True, drop_last=True)
    epoch_step_num = len(train_dataset) // batch_size

    """
    Setup optimizer
    """
    optimizer = torch.optim.Adam([
        {'params': net.module.backbone.parameters(),
         'lr': learning_rate,
         'weight_decay': weight_decay},
        {'params': net.module.coarse_pyramid_detection.parameters(),
         'lr': learning_rate,
         'weight_decay': weight_decay}
    ])
    scheduler = WarmupMultiEpochLR(optimizer, epoch_step_num, 1, [12, 18])

    """
    Start training
    """
    start_epoch = resume_training(resume, net, optimizer)

    for i in range(start_epoch, max_epoch + 1):
        print(f"epoch: {i}/{max_epoch}, LR: {optimizer.param_groups[0]['lr']}")
        run_one_epoch(i, net, optimizer, train_data_loader, len(train_dataset) // batch_size)
        scheduler.step()
        