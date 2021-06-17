import argparse
import multiprocessing
import os
import os.path as osp
import json
import numpy as np
import scipy.interpolate
from mmcv import dump, load
import statistics

def pool_feature_fun(data, num_proposals=200, num_sample_bins=3, pool_type='mean'):
    if len(data) == 1:
        return np.concatenate([data] * num_proposals)
        
    x_range = list(range(len(data)))
    f   = scipy.interpolate.interp1d(x_range, data, axis=0)
    eps = 1e-4
    start, end  = eps, len(data) - 1 - eps
    anchor_size = (end - start) / num_proposals
    ptr         = start
    feature     = []
    for i in range(num_proposals):
        x_new = [ptr + i / num_sample_bins * anchor_size for i in range(num_sample_bins)]
        y_new = f(x_new)
        if pool_type == 'mean':
            y_new = np.mean(y_new, axis=0)
        elif pool_type == 'max':
            y_new = np.max(y_new, axis=0)
        else:
            raise NotImplementedError('Unsupported pool type')
        feature.append(y_new)
        ptr += anchor_size
    feature  = np.stack(feature)
    return feature

# 317 
# 2678
# 1323

# 320 
# 3560
# 1439.8035143769969

# data_json = json.load(open("train.json", 'r'))

# feat_nums = []
# for file in data_json.keys():
    # file_path = file + '.npy'
    # save_path = file + '.csv'
    
    # raw_feature = np.load(file_path)
    # raw_feature = raw_feature.squeeze(axis=2).squeeze(axis=1)
    
    # pool_feature = pool_feature_fun(raw_feature)
    # print(raw_feature.shape, pool_feature.shape)
    
    # pool_feature = pool_feature.tolist()
    # lines = []
    # line0 = ','.join([f'f{i}' for i in range(1024)])
    # lines.append(line0)
    # for line in pool_feature:
        # lines.append(','.join([f'{x:.4f}' for x in line]))
        
    # with open(save_path, 'w') as fp:
        # fp.write('\n'.join(lines))
    
    
data_json = json.load(open("trainval.json", 'r'))

feat_nums = []
for file in data_json.keys():
    file_path = file + '.npy'
    save_path = 'tsm_features/' + file.split('/')[-1] + '.csv'
    
    raw_feature = np.load(file_path)
    raw_feature = raw_feature.squeeze(axis=2).squeeze(axis=1)
    
    pool_feature = pool_feature_fun(raw_feature)
    print(raw_feature.shape, pool_feature.shape)
    
    pool_feature = pool_feature.tolist()
    lines = []
    line0 = ','.join([f'f{i}' for i in range(1024)])
    lines.append(line0)
    for line in pool_feature:
        lines.append(','.join([f'{x:.4f}' for x in line]))
        
    with open(save_path, 'w') as fp:
        fp.write('\n'.join(lines))
            