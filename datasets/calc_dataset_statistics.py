import os
import pandas as pd
from PIL import Image

def compute_dancetrack_stats(dataset_path):
    """
    Compute dataset statistics for DanceTrack format across all splits.

    Args:
        dataset_path (str): Path to dataset root (contains train/val/test).

    Returns:
        dict: Statistics dictionary.
    """
    splits = ["train", "val", "test"]
    num_sequences = 0
    num_frames_total = 0
    seq_lengths = []
    num_bboxes_total = 0
    num_tracks_total = 0
    max_tracks_in_seq = 0

    min_width, min_height = float("inf"), float("inf")
    max_width, max_height = 0, 0

    for split in splits:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            continue

        seqs = sorted(os.listdir(split_path))
        num_sequences += len(seqs)

        for seq in seqs:
            seq_path = os.path.join(split_path, seq)

            # --- Ground truth ---
            gt_file = os.path.join(seq_path, "gt/gt.txt")
            if not os.path.exists(gt_file):
                continue

            df = pd.read_csv(gt_file, header=None)
            try:
                df.columns = [
                    "frame", "id", "bb_left", "bb_top", "bb_width", "bb_height",
                    "x", "y", "z"
                ]
            except:
                df.columns = [
                    "frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf",
                    "x", "y", "z"
                ]

            # sequence stats
            seq_len = df["frame"].max()
            seq_lengths.append(seq_len)
            num_frames_total += seq_len

            # bounding boxes
            num_bboxes_total += len(df)

            # tracks
            tracks = df["id"].unique()
            num_tracks = len(tracks)
            num_tracks_total += num_tracks
            max_tracks_in_seq = max(max_tracks_in_seq, num_tracks)

            # --- Resolution ---

            img_dir = os.path.join(seq_path, "img1")


            if os.path.exists(img_dir):
                img_files = sorted(os.listdir(img_dir))
                if img_files:
                    first_img = os.path.join(img_dir, img_files[0])
                    with Image.open(first_img) as img:
                        w, h = img.size
                        min_width = min(min_width, w)
                        min_height = min(min_height, h)
                        max_width = max(max_width, w)
                        max_height = max(max_height, h)

    stats = {
        "num_sequences": num_sequences,
        "num_frames_total": num_frames_total,
        "min_seq_length": min(seq_lengths) if seq_lengths else 0,
        "max_seq_length": max(seq_lengths) if seq_lengths else 0,
        "num_bboxes_total": num_bboxes_total,
        "num_tracks_total": num_tracks_total,
        "max_tracks_in_seq": max_tracks_in_seq,
        "min_resolution": (min_width, min_height) if min_width < float("inf") else None,
        "max_resolution": (max_width, max_height) if max_width > 0 else None,
    }
    return stats


if __name__ == "__main__":
    dataset_path = "/mnt/vast-nhr/projects/nib00021/data/public/gmot-40-animal"
    stats = compute_dancetrack_stats(dataset_path)
    print("\nDataset statistics (combined train/val/test):")
    for k, v in stats.items():
        print(f"{k}: {v}")
