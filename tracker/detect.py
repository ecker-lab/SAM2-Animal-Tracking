import os
import torch
import sys
import torchvision
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path
import os

from utils.misc import custom_collate_fn, ImageDataset
from utils.box_ops import preprocess_coco_targets
from detector import Detector


config = json.loads(sys.argv[1])

# setup paths
dataset_dir = os.path.join(
    config['DATA_ROOT'], config['DATASET'], config['DATA_SPLIT'])
output_dir = os.path.join(
    config['OUTPUT_DIR'], config['DATASET'], config['DATA_SPLIT'])
output_dir = os.path.join(output_dir, config['DETECTOR']['MODEL'])
Path(output_dir).mkdir(
    parents=True, exist_ok=True)

# setup model
model = Detector(config['DETECTOR'], out_box_format='xywh')

# create metric
metric = MeanAveragePrecision(iou_type="bbox", box_format='xywh')

# iterate over all sequences in given dataset split
sequences = os.listdir(dataset_dir)
sequences.sort()
for seq in sequences[:]:
    print('-----------------------')
    print(f'Start detecting for sequence "{seq}"')
    print('-----------------------')

    # paths
    video_dir = os.path.join(dataset_dir, seq, config['IMG_DIR'])
    gt_dir = os.path.join(dataset_dir, seq, 'gt', 'gt_det.json')

    eval_dets = os.path.exists(gt_dir)

    images = os.listdir(video_dir)
    images.sort()

    # create dataset and dataloader to support batch inference
    if eval_dets:
        seq_dataset = torchvision.datasets.CocoDetection(
            video_dir, gt_dir)
    else:
        seq_dataset = ImageDataset(video_dir)

    seq_dataloader = torch.utils.data.DataLoader(
        seq_dataset, batch_size=config['DETECTOR']['BATCH_SIZE'], collate_fn=custom_collate_fn)

    if config['DETECTOR']['LOAD_DETS']:
        model.load_dets(output_det_file = os.path.join(
            output_dir,seq + '.pt'))

    if config['SAVE_DETS']:
        detections_list = []

    for i, (imgs, targets) in enumerate(tqdm(seq_dataloader)):

        detections = model(imgs, i)

        if config['SAVE_DETS']:
            detections_list += detections

        if eval_dets:
            target = preprocess_coco_targets(
                targets, num_frames=len(detections))
            # labels are expected to start with 1 instead of zeroes for detection eval
            detections = [{k: (v + 1 if k == 'labels' else v) for k, v in d.items()} for d in detections]
            metric.update(detections, target)

    # print metric after every sequence
    if eval_dets:
        print(metric.compute())

    # saving detections to file
    if config['SAVE_DETS']:
        output_det_file = os.path.join(
            output_dir,seq + '.pt')
        torch.save(detections_list, output_det_file)

# calculate metrics over all images for each model
if eval_dets:
    metric_dict = metric.compute()
    print(metric_dict)
    metric.reset()
    metric_dict = {m: n.tolist()
                   for m, n in zip(metric_dict, metric_dict.values())}

    # save metrics to file
    json.dump(metric_dict, open(os.path.join(
        output_dir, os.path.basename(f"{config['DETECTOR']['MODEL']}.json")), 'w+'))
