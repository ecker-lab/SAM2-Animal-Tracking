import os
import shutil
import json
import subprocess
from pathlib import Path
from configparser import ConfigParser
from PIL import Image

# Paths
panaf_root = '/mnt/vast-nhr/projects/nib00021/data/public/panaf/panaf500'
dancetrack_root = '/mnt/vast-nhr/projects/nib00021/data/public/panaf500_tracking'
splits = ["validation"]


def convert_annotations(anno_json_path, output_txt_path):
    with open(anno_json_path) as f:
        data = json.load(f)

    with open(output_txt_path, 'w') as f_out:
        for frame_info in data['annotations']:
            frame_id = frame_info['frame_id']
            for detection in frame_info.get('detections', []):
                x1, y1, x2, y2 = detection['bbox']
                track_id = detection['ape_id']
                width = x2 - x1
                height = y2 - y1
                # Format: frame, id, x, y, w, h, 1, -1, -1, -1
                f_out.write(
                    f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},1,-1,-1,-1\n")


def extract_frames(video_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / "%06d.jpg")

    # Extract frames using ffmpeg
    subprocess.run([
        "ffmpeg", "-i", str(video_path),
        "-q:v", "2",  # Quality setting
        output_pattern
    ], check=True)

    # Get image properties from the first frame
    first_image = next(output_dir.glob("*.jpg"))
    with Image.open(first_image) as img:
        img_width, img_height = img.size

    frame_count = len(list(output_dir.glob("*.jpg")))

    # Get frame rate via ffprobe
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    r_frame_rate = result.stdout.strip()

    # Convert frame rate string like "25/1" to float
    num, denom = map(int, r_frame_rate.split('/'))
    frame_rate = num / denom if denom else 1

    return img_width, img_height, frame_rate, frame_count


def convert_split(split):
    print(f"Converting split: {split}")
    anno_dir = Path(panaf_root) / "annotations" / split
    video_dir = Path(panaf_root) / "videos"
    output_dir = Path(dancetrack_root) / split

    for anno_file in anno_dir.glob("*.json"):
        seq_name = anno_file.stem
        video_file = video_dir / f"{seq_name}.mp4"
        seq_output_dir = output_dir / seq_name
        img_dir = seq_output_dir / "img1"
        gt_dir = seq_output_dir / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)

        # Extract frames and get video info
        img_width, img_height, frame_rate, seq_length = extract_frames(
            video_file, img_dir)

        # Convert annotations
        convert_annotations(anno_file, gt_dir / "gt.txt")


for split in splits:
    convert_split(split)
