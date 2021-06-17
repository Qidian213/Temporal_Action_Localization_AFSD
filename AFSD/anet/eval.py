import argparse
import numpy as np
from AFSD.evaluation.eval_detection import ANETdetection

parser = argparse.ArgumentParser()
parser.add_argument('output_json', type=str)
parser.add_argument('gt_json', type=str,
                    default='/data/Dataset/MMAction/Task2_Cross_Modal_Untrimmed_Action_Temporal_Localization/untrimmed/val_ASFD.json', nargs='?')
args = parser.parse_args()

tious = np.linspace(0.5, 0.95, 10)
anet_detection = ANETdetection(
    ground_truth_filename=args.gt_json,
    prediction_filename=args.output_json,
    subset='validation', tiou_thresholds=tious)
    
mAPs, average_mAP, ap = anet_detection.evaluate()

for (tiou, mAP) in zip(tious, mAPs):
    print("mAP at tIoU {} is {}".format(tiou, mAP))
print('Average mAP:', average_mAP)
