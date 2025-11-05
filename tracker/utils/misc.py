import os
from tqdm import tqdm
import yaml
import json
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import torch
from torch import Tensor
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
from torchvision.ops import box_convert

from .box_ops import mask_to_xyxy


def save_images(frame_names: list, video_segments: dict, video_dir: str, output_dir: str, detections: dict, save_vid=False) -> None:
    """ saves all images/videos with visualized predictions

    Args:
        frame_names (list): chronological list of frame names
        video_segments (dict): frame-wise segmentation mask
        video_dir (str): current video directory
        output_dir (str): output directory
        detections (dict): frame-wise detections
        save_vid (bool, optional): save video in addition to images. Defaults to False.
    """
    seq_name = os.path.basename(os.path.dirname(video_dir))
    out_img_dir = os.path.join(output_dir, seq_name)

    for out_frame_idx in tqdm(range(len(frame_names))):

        image = Image.open(os.path.join(
            video_dir, frame_names[out_frame_idx])).convert('RGBA')

        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        drawing = ImageDraw.Draw(overlay)

        for out_obj_id, out_mask in video_segments[out_frame_idx].items():

            # filter out filler masks, as sam2 always needs to track atleast one object
            if out_obj_id < 0:
                continue


            cmap = plt.get_cmap("tab20")

            cmap_idx = 0 if out_obj_id is None else out_obj_id
            color = 255 * np.array([*cmap(cmap_idx)[:3], 0.6])
            color = (int(color[0].item()), int(color[1].item()), int(
                color[2].item()), int(color[3]))
            mask_image = out_mask.reshape(out_mask.shape[-2:])
            mask_image = Image.fromarray(
                (255 * mask_image).astype('uint8'), mode='L')
            drawing.bitmap((0, 0), mask_image, fill=color)

        image = Image.alpha_composite(image, overlay)

        draw = ImageDraw.Draw(image)
        for i, box in enumerate(detections[out_frame_idx]['boxes']):

            # filter out filler detections
            if detections[out_frame_idx]['scores'][i] == 0:
                continue

            draw.text((box[0] + 5, box[1] + 5),
                      str(round(detections[out_frame_idx]['scores'][i].cpu().item(), 2)), fill=(0, 0, 0))
            draw.rectangle([(box[0], box[1]), (box[2], box[3])],
                           outline=(0, 0, 0), width=2)

        Path(out_img_dir).mkdir(parents=True, exist_ok=True)

        image.save(os.path.join(out_img_dir, os.path.splitext(
            frame_names[out_frame_idx])[0] + '.png'), format="PNG")

    if save_vid:
        Path(os.path.join(output_dir, 'videos')).mkdir(
            parents=True, exist_ok=True)
        os.system(
            f"ffmpeg -y -framerate 30 -pattern_type glob -i '{out_img_dir}/*.png' -c:v libx264 -pix_fmt yuv420p {output_dir}/videos/{seq_name}.mp4")
        

def save_preds(video_segments: dict, output_dir: str, seq_name: str) -> None:
    """ saves tracking predictions in dancetrack/mot17 format

    Args:
        video_segments (dict): frame-wise segmentation masks
        output_dir (str): where to save the output
        seq_name (str): name of current sequence
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / f'{seq_name}.txt'
    output_file.touch(exist_ok=True)
    with open(output_file, 'w+') as f:
        for out_frame_idx in range(len(video_segments)):
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                # filter out filler masks, as sam2 always needs to track atleast one object
                if out_obj_id < 0:
                    continue
                box_xyxy = mask_to_xyxy(out_mask)[0, :]
                box_list = box_convert(torch.tensor(
                    box_xyxy).unsqueeze(0), 'xyxy', 'xywh').squeeze().tolist()
                if all(v == 0 for v in box_list):
                    continue
                line = [str(b) for b in box_list]
                line.insert(0, str(out_frame_idx + 1))
                line.insert(1, str(out_obj_id))
                line = ','.join(line)
                f.write(line + ',1,1,1\n')


def save_preds_bdd100k(frame_names: list, class_names: list, video_segments: dict, track_labels: dict, output_dir: str, seq_name: str) -> None:
    """saves predictions in bdd100k format

    Args:
        frame_names (list): list of frame names
        class_names (list): list of class names
        video_segments (dict): mask predictions 
        track_labels (dict): class labels corresponding to class ids
        output_dir (str): output path
        seq_name (str): video sequence name
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{seq_name}.json")

    json_output = []

    for i, frame in enumerate(frame_names):
        frame_id = i
        frame_name = frame 
        labels = []

        for track_id, masks in video_segments[frame_id].items():
            if track_id < 0:
                continue
            class_id = int(track_labels.get(track_id, 'unknown'))
            class_label = class_names[class_id]
            box = mask_to_xyxy(masks)[0, :]
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            labels.append({
                "id": str(track_id),
                "category": class_label,
                "box2d": {"x1": str(x1), "y1": str(y1), "x2": str(x2), "y2": str(y2)}
            })

        frame_entry = {
            "videoName": seq_name,
            "name": frame_name,
            "frameIndex": frame_id,
            "labels": labels
        }
        json_output.append(frame_entry)

    # Save JSON file
    with open(output_file, 'w') as f:
        json.dump(json_output, f, indent=4)


def reorder_coco_ground_truth(flat_targets:list, image_ids_in_batch:list) -> list:
    """
    Reorders flat COCO ground-truth annotations into per-image format.

    Args:
        flat_targets (list): Flat list of annotation dicts.
        image_ids_in_batch (list[int]): List of image_ids in the batch in the order the images came.

    Returns:
        list: Reordered annotations, grouped per image.
    """
    # Group annotations by image_id
    grouped = defaultdict(list)
    for ann in flat_targets:
        grouped[ann['image_id']].append(ann)

    # Reorder according to the order of images
    ordered_annotations = [grouped.get(image_id, [])
                           for image_id in image_ids_in_batch]
    return ordered_annotations


def custom_collate_fn(batch:tuple) -> tuple[list,list]:
    """collates samples as lists.

    Args:
        batch (tuple): tuple of images and annotations

    Returns:
        tuple[list,list]: lists of images and targets
    """
    images = [item[0] for item in batch]
    annotations = [item[1] for item in batch]  # list of lists

    if all(x is None for x in annotations):
        return images, annotations

    flat_annotations = [ann for sublist in annotations for ann in sublist]
    image_ids = [ann[0]['image_id'] for ann in annotations if len(
        ann) > 0]  # assume at least 1 ann per image

    reordered_targets = reorder_coco_ground_truth(
        flat_annotations, image_ids)

    return images, reordered_targets


def yaml_to_dict(path: str) -> dict:
    """
    Read a yaml file into a dict.

    Args:
        path (str): The path of yaml file.

    Returns:
        A dict.
    """
    with open(path) as f:
        return yaml.load(f.read(), yaml.FullLoader)


class ImageDataset(torch.utils.data.Dataset):
    """minimalistic dataset of all images in a sequence
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.files = [f for f in os.listdir(root) if f.endswith(".jpg")]
        self.files.sort()
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.files[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, None
    

def move_to(obj:Tensor | list | dict, device: torch.device) -> Tensor | list | dict:
    """moves all tensors in object to the given device

    Args:
        obj (Tensor | list | dict): object to move to target device
        device (torch.device): target device

    Returns:
        Tensor | list | dict: object moved to target device
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")
    

def mask_dict(dict:dict, mask:Tensor) -> Tensor:
    """masks all tensors in dict

    Args:
        dict (dict): dict to be masked
        mask (Tensor): mask

    Returns:
        Tensor: dict with masked tensors
    """
    return {key: value[mask] for key, value in dict.items()}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class PresetAction(argparse.Action):
    def __init__(self, option_strings, dest, presets, **kwargs):
        self.presets = presets
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values not in self.presets:
            parser.error(f"Unknown preset '{values}' for {self.dest}")
        # Set the preset name
        setattr(namespace, self.dest, values)
        # Dynamically set all keys from the preset directly in args
        for key, val in self.presets[values].items():
            setattr(namespace, key, val)


def nest_config(args):
    """Turn flat dotted args into a nested dict."""
    config = {}
    for key, value in vars(args).items():
        parts = key.split(".")
        d = config
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return config


def process_text_labels(detections, reference_labels):
    """
    Filters detection dicts to only keep ['labels', 'boxes', 'scores'].
    Converts string labels to integer indices based on reference_labels.
    Removes unmatched entries from all fields consistently.
    """
    valid_keys = ['labels', 'boxes', 'scores']
    cleaned = []

    for det in detections:
        # keep only relevant keys
        det = {k: v for k, v in det.items() if k in valid_keys}
        if 'labels' not in det:
            continue

        labels = det['labels']

        # Case 1: labels are list of strings
        if isinstance(labels, list) and all(isinstance(l, str) for l in labels):
            keep_indices = []
            label_indices = []

            for i, label in enumerate(labels):
                if label in reference_labels:
                    keep_indices.append(i)
                    label_indices.append(reference_labels.index(label))

            # if no matches, skip this detection
            if not keep_indices:
                continue

            det['labels'] = torch.tensor(label_indices, dtype=torch.long, device=det['boxes'].device)

            # filter boxes and scores accordingly
            for k in ['boxes', 'scores']:
                if k in det and torch.is_tensor(det[k]):
                    det[k] = det[k][keep_indices]

        # Case 2: labels already tensor
        elif torch.is_tensor(labels):
            det['labels'] = labels.clone()

        else:
            continue

        cleaned.append(det)

    return cleaned
