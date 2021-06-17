import os
import sys
from multiprocessing import Pool

def extract_frame(vid_item):
    src_full_path, out_flow_path = vid_item

    os.system(f'/home/user_zzg/app/bin/denseflow {src_full_path} -b=20 -a=tvl1 -s=1 -o={out_flow_path} -v')
    
    print(f'{src_full_path} frame flow done')
    sys.stdout.flush()
    return True

if __name__ == '__main__':
    video_dir = '/data/Dataset/MMAction/Task2_Cross_Modal_Untrimmed_Action_Temporal_Localization/test/video_112_2304/'
    flow_dir  = '/data/Dataset/MMAction/Task2_Cross_Modal_Untrimmed_Action_Temporal_Localization/test/video_112_2304_flow/'

    src_paths = []
    dst_paths = []
    for file in os.listdir(video_dir):
        video_path = video_dir + file
        src_paths.append(video_path)
        dst_paths.append(flow_dir)
        
    pool = Pool(6)
    pool.map(extract_frame, zip(src_paths, dst_paths))