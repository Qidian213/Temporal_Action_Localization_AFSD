import os
import multiprocessing as mp
import argparse
import cv2
import json

parser = argparse.ArgumentParser()
parser.add_argument('thread_num', type=int)
parser.add_argument('--video_dir', type=str, default='/data/Dataset/MMAction/Task2_Cross_Modal_Untrimmed_Action_Temporal_Localization/untrimmed')
parser.add_argument('--output_dir', type=str, default='/data/Dataset/MMAction/Task2_Cross_Modal_Untrimmed_Action_Temporal_Localization/untrimmed/video_112_2304')
parser.add_argument('--resolution', type=str, default='112x112')
parser.add_argument('--max_frame', type=int, default=2304)
args = parser.parse_args()

thread_num = args.thread_num
video_dir  = args.video_dir
output_dir = args.output_dir
resolution = args.resolution
max_frame  = args.max_frame

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data_json = json.load(open("/data/Dataset/MMAction/Task2_Cross_Modal_Untrimmed_Action_Temporal_Localization/untrimmed/trainval.json", 'r'))
files = []
for key in data_json.keys():
    files.append(key + '.mp4')

# pfiles = sorted(os.listdir(video_dir))
# files = []
# for file in pfiles:
    # if('.mp4' in file):
        # files.append(file)
    
def sub_processor(pid, files):
    for file in files[:]:
        file_name   = file.split('/')[-1].replace('.mp4', '') 
        target_file = os.path.join(output_dir, file_name + '.mp4')
        
        if os.path.exists(target_file):
            print('{} exists, skip.'.format(target_file))
            continue
        cap       = cv2.VideoCapture(os.path.join(video_dir, file))
        max_fps   = cap.get(cv2.CAP_PROP_FPS)
        frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        ratio     = min(max_frame * 1.0 / frame_num, 1.0)
        target_fps = max_fps * ratio
        cmd = 'ffmpeg -v quiet -i {} -qscale 0 -r {} -s {} -y {}'.format(
            os.path.join(video_dir, file),
            target_fps,
            resolution,
            target_file
        )
        os.system(cmd)

processes = []
video_num = len(files)
per_process_video_num = video_num // thread_num

for i in range(thread_num):
    if i == thread_num - 1:
        sub_files = files[i * per_process_video_num:]
    else:
        sub_files = files[i * per_process_video_num: (i + 1) * per_process_video_num]
    p = mp.Process(target=sub_processor, args=(i, sub_files))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
