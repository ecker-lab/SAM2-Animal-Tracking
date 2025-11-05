import os
import shutil
from generate_trackeval_files import gen_trackeval_files

def convert_animaltrack_to_dancetrack(src_root, dst_root, test_seqs):
    """
    Convert AnimalTrack dataset to DanceTrack format.
    
    Args:
        src_root (str): Path to Whole_AnimalTrack (contains frames_all/ and gt_all/).
        dst_root (str): Output path for converted dataset (DanceTrack style).
        test_seqs (list[str]): List of sequence names to put into 'test'.
    """
    frames_root = os.path.join(src_root, "frames_all")
    gt_root = os.path.join(src_root, "gt_all")

    # Create output dirs
    train_dir = os.path.join(dst_root, "train")
    test_dir = os.path.join(dst_root, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Iterate over sequences
    for seq_name in os.listdir(frames_root):
        seq_path = os.path.join(frames_root, seq_name)
        if not os.path.isdir(seq_path):
            continue

        # Decide split
        split = "test" if seq_name in test_seqs else "train"
        out_seq_path = os.path.join(dst_root, split, seq_name)
        img_out = os.path.join(out_seq_path, "img1")
        gt_out = os.path.join(out_seq_path, "gt")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(gt_out, exist_ok=True)

        # Copy frames
        for fname in sorted(os.listdir(seq_path)):
            if fname.endswith(".jpg"):
                shutil.copy(os.path.join(seq_path, fname), os.path.join(img_out, fname))

        # Copy and rename GT
        gt_file = os.path.join(gt_root, f"{seq_name}_gt.txt")
        if os.path.exists(gt_file):
            shutil.copy(gt_file, os.path.join(gt_out, "gt.txt"))
        else:
            print(f"Warning: no GT file for {seq_name}")

        print(f"Converted {seq_name} -> {split}")

if __name__ == "__main__":
    src_root = "/mnt/vast-nhr/projects/nib00021/data/public/Whole_AnimalTrack"
    dst_root = "/mnt/vast-nhr/projects/nib00021/data/public/AnimalTrack_dancetrack"
    test_seqs = ["chicken_1","chicken_2","deer_1","deer_2","deer_3","dolphin_1","dolphin_2","dolphin_3","duck_1","duck_2","duck_3","goose_1","goose_2","goose_3","horse_1","horse_2","horse_3","penguin_1","penguin_2","penguin_3","pig_1","pig_2","rabbit_1","rabbit_2","zebra_1","zebra_2"]

    # convert file structure
    # convert_animaltrack_to_dancetrack(src_root, dst_root, test_seqs)

    # generate seqmap and seqinfo file
    gen_trackeval_files('/mnt/vast-nhr/projects/nib00021/data/public/AnimalTrack_dancetrack', ['train','test'])