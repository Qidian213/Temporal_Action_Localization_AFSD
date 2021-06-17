import torch
import torch.nn as nn
import os
import numpy as np
import tqdm
import json
from AFSD.common import videotransforms
from AFSD.common.anet_dataset import get_video_info, load_json
from AFSD.anet.BDNet import BDNet
from AFSD.common.segment_utils import softnms_v2
from AFSD.common.config import config

import multiprocessing as mp
import threading

num_classes = config['dataset']['num_classes']
conf_thresh = config['testing']['conf_thresh']
top_k       = config['testing']['top_k']
nms_thresh  = config['testing']['nms_thresh']
nms_sigma   = config['testing']['nms_sigma']
clip_length = config['dataset']['testing']['clip_length']
stride      = config['dataset']['testing']['clip_stride']
crop_size   = config['dataset']['testing']['crop_size']
checkpoint_path = config['testing']['checkpoint_path']
json_name    = 'untrimmed_submission.json'
output_path  = config['testing']['output_path']
ngpu         = config['ngpu']
softmax_func = True
if not os.path.exists(output_path):
    os.makedirs(output_path)

thread_num = ngpu
global result_dict
result_dict = mp.Manager().dict()

processes = []
lock = threading.Lock()

name2id = {'standing': 1, 
    'crouching': 2, 
    'walking': 3, 
    'running': 4, 
    'checking_time': 5, 
    'waving_hand': 6, 
    'using_phone': 7, 
    'talking_on_phone': 8, 
    'kicking': 9, 
    'pointing': 10, 
    'throwing': 11, 
    'jumping': 12, 
    'exiting': 13, 
    'entering': 14, 
    'setting_down': 15, 
    'talking': 16, 
    'opening': 17, 
    'closing': 18, 
    'carrying': 19, 
    'loitering': 20, 
    'transferring_object': 21, 
    'looking_around': 22, 
    'pushing': 23, 
    'pulling': 24, 
    'picking_up': 25, 
    'fall': 26, 
    'carrying_light': 27, 
   # 'Carrying_light': 27, 
    'carrying_heavy': 28, 
    'sitting_down': 29, 
    'using_pc': 30, 
    'drinking': 31, 
    'pocket_out': 32, 
    'pocket_in': 33, 
    'sitting': 34, 
    'using_phone_desk': 35, 
    'talking_on_phone_desk': 36, 
    'standing_up': 37}

id2name = {value:key for key, value in name2id.items()}

test_ASFD_Submit = '/data/Dataset/MMAction/Task2_Cross_Modal_Untrimmed_Action_Temporal_Localization/test/test_ASFD_Submit.json'
video_mp4_path   = '/data/Dataset/MMAction/Task2_Cross_Modal_Untrimmed_Action_Temporal_Localization/test/video_112_2304_npy'
#video_mp4_path   = '/data/Dataset/MMAction/Task2_Cross_Modal_Untrimmed_Action_Temporal_Localization/test/video_112_2304_flow_npy'
video_infos   = get_video_info(test_ASFD_Submit, subset='testing')
mp4_data_path = video_mp4_path

if softmax_func:
    score_func = nn.Softmax(dim=-1)
else:
    score_func = nn.Sigmoid()

centor_crop = videotransforms.CenterCrop(crop_size)
resize      = videotransforms.ResizeClip(crop_size)

video_list = list(video_infos.keys())
video_num  = len(video_list)
per_thread_video_num = video_num // thread_num

def sub_processor(lock, pid, video_list):
    text = 'processor %d' % pid
    with lock:
        progress = tqdm.tqdm(
            total=len(video_list),
            position=pid,
            desc=text,
            ncols=0
        )
    channels = config['model']['in_channels']
    torch.cuda.set_device(pid)
    net = BDNet(in_channels=channels,training=False)
    net.load_state_dict(torch.load(checkpoint_path))
    net.eval().cuda()

    for video_name in video_list:
        sample_count = video_infos[video_name]['frame_num']
        sample_fps   = video_infos[video_name]['fps']
        duration     = video_infos[video_name]['duration']

        offsetlist = [0]

        data   = np.load(os.path.join(mp4_data_path, video_name + '.npy'))
        frames = data
        
        frames = np.transpose(frames, [3, 0, 1, 2])
        data = centor_crop(frames)
        data = torch.from_numpy(data.copy())

        output = []
        for cl in range(num_classes):
            output.append([])
        res = torch.zeros(num_classes, top_k, 3)

        for offset in offsetlist:
            clip = data[:, offset: offset + clip_length]
            clip = clip.float()
            if clip.size(1) < clip_length:
                tmp = torch.ones(
                    [clip.size(0), clip_length - clip.size(1), crop_size, crop_size]).float() * 127.5
                clip = torch.cat([clip, tmp], dim=1)
            clip = clip.unsqueeze(0).cuda()
            clip = (clip / 255.0) * 2.0 - 1.0
            with torch.no_grad():
                output_dict = net(clip)

            loc, conf, priors = output_dict['loc'], output_dict['conf'], output_dict['priors']
            prop_loc, prop_conf = output_dict['prop_loc'], output_dict['prop_conf']
            center = output_dict['center']
            loc = loc[0]
            conf = score_func(conf[0])
            prop_loc = prop_loc[0]
            prop_conf = score_func(prop_conf[0])
            center = center[0].sigmoid()

            pre_loc_w = loc[:, :1] + loc[:, 1:]
            loc = 0.5 * pre_loc_w * prop_loc + loc
            decoded_segments = torch.cat(
                [priors[:, :1] * clip_length - loc[:, :1],
                 priors[:, :1] * clip_length + loc[:, 1:]], dim=-1)
            decoded_segments.clamp_(min=0, max=clip_length)

            conf = (conf + prop_conf) / 2.0
            conf = conf * center
            conf = conf.view(-1, num_classes).transpose(1, 0)
            conf_scores = conf.clone()

            for cl in range(1, num_classes):
                c_mask = conf_scores[cl] > 1e-9
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_segments)
                segments = decoded_segments[l_mask].view(-1, 2)
                segments = (segments + offset) / sample_fps
                segments = torch.cat([segments, scores.unsqueeze(1)], -1)

                output[cl].append(segments)

        sum_count = 0
        for cl in range(1, num_classes):
            if len(output[cl]) == 0:
                continue
            tmp = torch.cat(output[cl], 0)
            tmp, count = softnms_v2(tmp, sigma=nms_sigma, top_k=top_k, score_threshold=1e-9)
            res[cl, :count] = tmp
            sum_count += count

        flt = res.contiguous().view(-1, 3)
        flt = flt.view(num_classes, -1, 3)
        proposal_list = []
        for cl in range(1, num_classes):
            class_name = id2name[cl]
            tmp = flt[cl].contiguous()
            tmp = tmp[(tmp[:, 2] > 0).unsqueeze(-1).expand_as(tmp)].view(-1, 3)
            if tmp.size(0) == 0:
                continue
            tmp = tmp.detach().cpu().numpy()
            for i in range(tmp.shape[0]):
                tmp_proposal = {}
                start_time = max(0, float(tmp[i, 0]))
                end_time = min(duration, float(tmp[i, 1]))
                if end_time <= start_time:
                    continue

                tmp_proposal['label'] = class_name
                tmp_proposal['score'] = float(tmp[i, 2])# * cuhk_score_1
                tmp_proposal['segment'] = [start_time, end_time]
                proposal_list.append(tmp_proposal)

        result_dict[video_name] = proposal_list
        with lock:
            progress.update(1)
    with lock:
        progress.close()

for i in range(thread_num):
    if i == thread_num - 1:
        sub_video_list = video_list[i * per_thread_video_num:]
    else:
        sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
    p = mp.Process(target=sub_processor, args=(lock, i, sub_video_list))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

##### post process
#output_dict = {"results": dict(result_dict)}
output_dict = {"results": dict(result_dict)}

with open(os.path.join(output_path, json_name), "w") as out:
    json.dump(output_dict, out)
