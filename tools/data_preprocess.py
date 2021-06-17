import io
import os
import json
import statistics
import decord
import numpy as np
from mmcv.fileio import FileClient
file_client = FileClient('disk')

# {'standing': 471, 
# 'crouching': 468, 
# 'walking': 468, 
# 'running': 468, 
# 'checking_time': 468, 
# 'waving_hand': 468, 
# 'using_phone': 468, 
# 'talking_on_phone': 468, 
# 'kicking': 468, 
# 'pointing': 468, 
# 'throwing': 468, 
# 'jumping': 472, 
# 'exiting': 464, 
# 'entering': 468, 
# 'setting_down': 472, 
# 'talking': 464, 
# 'opening': 472, 
# 'closing': 472, 
# 'carrying': 468, 
# 'loitering': 476, 
# 'transferring_object': 464, 
# 'looking_around': 468, 
# 'pushing': 468, 
# 'pulling': 468, 
# 'picking_up': 464, 
# 'fall': 472, 
# 'carrying_light': 236, 
# 'carrying_heavy': 240, 
# 'Carrying_light': 4, 
# 'sitting_down': 158, 
# 'using_pc': 158, 
# 'drinking': 158, 
# 'pocket_out': 158, 
# 'pocket_in': 158, 
# 'sitting': 158, 
# 'using_phone_desk': 158, 
# 'talking_on_phone_desk': 158, 
# 'standing_up': 158}

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
    'Carrying_light': 27, 
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

train_anns = {}

ann_dir   = '/data/Dataset/MMAction/Task2_Cross_Modal_Untrimmed_Action_Temporal_Localization/untrimmed/annotation/'
video_dir = '/data/Dataset/MMAction/Task2_Cross_Modal_Untrimmed_Action_Temporal_Localization/untrimmed/video/'

f_scene_train = open("splits/untrimmed_x-session_train.txt", "r")
for line in f_scene_train.readlines():
    line = line.strip('\n')
    
    line     = line.split(',')
    st_time  = line[0].split('-')[1]
    et_time  = line[1].split('-')[1]
    st_time  = float(st_time[:2])*60*60 + float(st_time[2:4])*60 + float(st_time[4:6])
    et_time  = float(et_time[:2])*60*60 + float(et_time[2:4])*60 + float(et_time[4:6])
    
    video_path   = video_dir + line[2] + line[1] + '.mp4'
    feature_path = video_dir + line[2] + line[1] + '.npy'
    
    if(os.path.isfile(video_path)):

        results={}
        results['filename'] = video_path
        file_obj  = io.BytesIO(file_client.get(results['filename']))
        container = decord.VideoReader(file_obj, num_threads=1)
    
    #    features = np.load(feature_path)
    
        ann_path = ann_dir + 'trainval/' + line[2][14:]
        
        ann_item = []
        files = os.listdir(ann_path)
        for file in files:
            ann_file = ann_path + file
            f_ann = open(ann_file, "r")
            for f_line in f_ann.readlines():
                f_line = f_line.strip('\n')
                f_line = f_line.split('-')
                
                ast_time = f_line[0].split(' ')[1]
                aet_time = f_line[1].split(' ')[1]
                label    = f_line[2]

                ast_time  = float(ast_time[:2])*60*60 + float(ast_time[3:5])*60 + float(ast_time[6:])
                aet_time  = float(aet_time[:2])*60*60 + float(aet_time[3:5])*60 + float(aet_time[6:])

                act_st = ast_time - st_time
                act_et = aet_time - st_time
                
                ann_line = {}
                ann_line["segment"] = [act_st, act_et]
                ann_line["label"] = name2id[label]
                ann_item.append(ann_line)
                
                #print(act_st, act_et, act_et-act_st)
        
        item_key = 'video/' + line[2] + line[1]
        train_anns[item_key] = {}
        train_anns[item_key]["duration_second"] = et_time - st_time
        train_anns[item_key]["duration_frame"]  = len(container)
        train_anns[item_key]["annotations"]     = ann_item
        train_anns[item_key]["feature_frame"]   = len(container)
        train_anns[item_key]["fps"]             = container.get_avg_fps()
        train_anns[item_key]["rfps"]            = container.get_avg_fps()

print(name2id)

json.dump(train_anns, open('train.json', 'w'), indent=4)
