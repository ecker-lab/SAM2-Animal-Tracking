import os
import shutil
from generate_trackeval_files import gen_trackeval_files

def convert_gmot40_to_dancetrack(src_root, dst_root, split="train"):
    """
    Convert GMOT-40 dataset into DanceTrack format.

    Args:
        src_root (str): Path to gmot-40 root (contains GenericMOT_JPEG_Sequence/ and track_label/).
        dst_root (str): Output path for converted dataset (DanceTrack-style).
        split (str): "train" or "test" (default: "train").
    """
    frames_root = os.path.join(src_root, "GenericMOT_JPEG_Sequence")
    labels_root = os.path.join(src_root, "track_label")

    out_split_dir = os.path.join(dst_root, split)
    os.makedirs(out_split_dir, exist_ok=True)

    for seq_name in os.listdir(frames_root):
        seq_path = os.path.join(frames_root, seq_name)
        img_in = os.path.join(seq_path, "img1")
        if not os.path.isdir(img_in):
            continue

        out_seq_dir = os.path.join(out_split_dir, seq_name)
        img_out = os.path.join(out_seq_dir, "img1")
        gt_out = os.path.join(out_seq_dir, "gt")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(gt_out, exist_ok=True)

        # Copy frames
        for fname in sorted(os.listdir(img_in)):
            if fname.lower().endswith(".jpg"):
                shutil.copy(os.path.join(img_in, fname),
                            os.path.join(img_out, fname))

        # Copy and rename GT
        label_file = os.path.join(labels_root, f"{seq_name}.txt")
        if os.path.exists(label_file):
            shutil.copy(label_file, os.path.join(gt_out, "gt.txt"))
        else:
            print(f"⚠️ Warning: no label file for {seq_name}")

        print(f"✅ Converted {seq_name} -> {out_seq_dir}")


def create_animal_subset(src_root, dst_root, keywords, split="train"):
    """
    Create an animal-only subset of a DanceTrack-style dataset (GMOT-40).

    Args:
        src_root (str): Path to GMOT-40 dataset in DanceTrack format.
        dst_root (str): Path to output subset (gmot-40-animal).
        keywords (list[str]): List of substrings to identify animal sequences.
        split (str): Which split to filter ("train" or "test").
    """
    src_split = os.path.join(src_root, split)
    dst_split = os.path.join(dst_root, split)
    os.makedirs(dst_split, exist_ok=True)

    seq_names = [d for d in os.listdir(src_split) if os.path.isdir(os.path.join(src_split, d))]

    for seq_name in seq_names:
        if any(keyword.lower() in seq_name.lower() for keyword in keywords):
            src_seq_path = os.path.join(src_split, seq_name)
            dst_seq_path = os.path.join(dst_split, seq_name)
            if os.path.exists(dst_seq_path):
                print(f"⚠️ Skipping {seq_name}, already exists in {dst_split}")
                continue
            shutil.copytree(src_seq_path, dst_seq_path)
            print(f"✅ Copied {seq_name} to {dst_split}")

def shift_gt_frames(root_dir, splits):
    """
    Shift frame numbers in all gt.txt files by +1 in a DanceTrack-style dataset.

    Args:
        root_dir (str): Path to dataset root (e.g. 'gmot-40').
    """
    for split in splits:  # adjust if you only have train
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            continue

        for seq in os.listdir(split_dir):
            gt_path = os.path.join(split_dir, seq, "gt", "gt.txt")
            if not os.path.isfile(gt_path):
                continue

            # Read and shift
            with open(gt_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split(",")
                if not parts or not parts[0].isdigit():
                    continue
                parts[0] = str(int(parts[0]) + 1)  # shift frame index
                new_lines.append(",".join(parts) + "\n")

            # Overwrite file
            with open(gt_path, "w") as f:
                f.writelines(new_lines)

            print(f"✅ Shifted frame indices in {gt_path}")


if __name__ == "__main__":
    src_root = "/mnt/vast-nhr/projects/nib00021/data/public/gmot-40"
    dst_root = "/mnt/vast-nhr/projects/nib00021/data/public/gmot-40_dancetrack"

    # convert file structure
    convert_gmot40_to_dancetrack(src_root, dst_root, split="test")

    # generate seqmap and seqinfo file
    gen_trackeval_files('/mnt/vast-nhr/projects/nib00021/data/public/gmot-40_dancetrack', ['test'])
    shift_gt_frames('/mnt/vast-nhr/projects/nib00021/data/public/gmot-40_dancetrack', ['test'])

    # create animal subset from gmot-40
    keywords = ["bird", "fish", "insect", "stock"]
    src_root = "/mnt/vast-nhr/projects/nib00021/data/public/gmot-40_dancetrack"
    dst_root = "/mnt/vast-nhr/projects/nib00021/data/public/gmot-40-animal"
    #create_animal_subset(src_root, dst_root, keywords, split="test")
    gen_trackeval_files('/mnt/vast-nhr/projects/nib00021/data/public/gmot-40-animal', ['test'])
    shift_gt_frames('/mnt/vast-nhr/projects/nib00021/data/public/gmot-40-animal', ['test'])