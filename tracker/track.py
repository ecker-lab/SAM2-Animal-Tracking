import json
import sys
import torch
import os

from sam2tracker import SAM2Tracker
from utils.misc import save_images, save_preds, save_preds_bdd100k

# Read arguments from the command line
video_dir = sys.argv[1]
seq = sys.argv[2]
config = json.loads(sys.argv[3])

# setup device and cuda configs
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
else:
    print('cuda not available')

# setup paths
dataset_dir = os.path.join(
    config['DATA_ROOT'], config['DATASET'], config['DATA_SPLIT'])
output_dir = os.path.join(
    config['OUTPUT_DIR'], config['DATASET'], config['DATA_SPLIT'], config['TEST_NAME'])

# init tracker
sam2tracker = SAM2Tracker(config, device)

# track sequence
video_segments, video_dets, frame_names, track_labels = sam2tracker.forward_sequence(
    video_dir)

# save images and predictions
if config['SAVE_IMGS']:
    print('Start saving images')
    save_images(frame_names, video_segments, video_dir,
                output_dir, video_dets, config['SAVE_VIDS'])

if config['SAVE_PREDS']:
    if config['DATASET'] == 'bdd100k':
        save_preds_bdd100k(frame_names, config['DETECTOR']['TEXT_PROMPT']
                           [0], video_segments, track_labels, output_dir, seq)
    else:
        save_preds(video_segments, output_dir, seq)
