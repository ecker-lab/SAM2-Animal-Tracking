import os
import xml.etree.ElementTree as ET
from pathlib import Path
import subprocess
import shutil


def extract_frames_ffmpeg(video_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-q:v", "2",
        "-start_number", "1",
        str(output_dir / "%06d.jpg")
    ]
    subprocess.run(cmd, check=True)


def parse_baboonland_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tracks = {}

    for track in root.findall("track"):
        track_id = int(track.attrib["id"])
        for box in track.findall("box"):
            if box.attrib["outside"] == "1":
                continue  # Skip boxes marked as outside
            frame_id = int(box.attrib["frame"]) + 1  # 1-based indexing
            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])
            width = xbr - xtl
            height = ybr - ytl

            tracks.setdefault(frame_id, []).append(
                [frame_id, track_id, xtl, ytl, width, height, 1, -1, -1, -1]
            )
    return tracks


def save_dancetrack_annotations(tracks_dict, output_txt_path):
    with open(output_txt_path, "w") as f:
        for frame_id in sorted(tracks_dict.keys()):
            for row in tracks_dict[frame_id]:
                row_str = ",".join([f"{x:.2f}" if isinstance(
                    x, float) else str(x) for x in row])
                f.write(row_str + "\n")


def convert_baboonland_to_dancetrack(baboonland_dir, dancetrack_dir):
    baboonland_dir = Path(baboonland_dir)
    dancetrack_dir = Path(dancetrack_dir)

    for split in ["train", "test"]:
        split_dir = baboonland_dir / split
        if not split_dir.exists():
            continue

        for seq_dir in split_dir.iterdir():
            if not seq_dir.is_dir():
                continue

            print(f"Processing {seq_dir}")
            xml_file = seq_dir / "tracks.xml"
            video_file = seq_dir / "video.mp4"

            output_seq_dir = dancetrack_dir / split / seq_dir.name
            img_dir = output_seq_dir / "img1"
            ann_dir = output_seq_dir
            ann_file = ann_dir / "gt" / "gt.txt"

            # Create directories
            (ann_dir / "gt").mkdir(parents=True, exist_ok=True)

            # Extract frames
            extract_frames_ffmpeg(video_file, img_dir)

            # Parse and save annotations
            tracks_dict = parse_baboonland_xml(xml_file)
            save_dancetrack_annotations(tracks_dict, ann_file)


convert_baboonland_to_dancetrack("data/public/baboonland_tracking",
                                 "data/public/baboonland_dancetrack")
