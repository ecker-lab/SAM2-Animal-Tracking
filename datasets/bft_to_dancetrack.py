import shutil
from pathlib import Path
from generate_trackeval_files import gen_trackeval_files



def convert_bft_to_dancetrack(src_root, dst_root, splits):
    # Iterate over sequences
    for split in splits:
        mot_ann_root = Path(src_root) / "annotations_mot" / split
        output_root = Path(dst_root) / split
        img_root = Path(src_root) / split
        for ann_file in mot_ann_root.glob("*.txt"):
            seq_name = ann_file.stem
            seq_img_folder = img_root / seq_name
            if not seq_img_folder.exists():
                print(f"No images found for {seq_name}, skipping.")
                continue

            # Create DanceTrack folder structure
            out_seq_folder = output_root / seq_name
            img1_folder = out_seq_folder / "img1"
            gt_folder = out_seq_folder / "gt"
            img1_folder.mkdir(parents=True, exist_ok=True)
            gt_folder.mkdir(parents=True, exist_ok=True)

            # Copy images
            for img_file in seq_img_folder.glob("*.jpg"):
                shutil.copy(img_file, img1_folder / img_file.name)

            # Copy annotation file
            shutil.copy(ann_file, gt_folder / "gt.txt")

            print(f"Converted sequence {seq_name} to DanceTrack format.")

if __name__ == "__main__":

    src_root = "/mnt/vast-nhr/projects/nib00021/data/public/BFT"
    dst_root = "/mnt/vast-nhr/projects/nib00021/data/public/BFT_dancetrack"

    # convert file structure
    convert_bft_to_dancetrack(src_root, dst_root, ['train','val','test'])

    # generate seqmap and seqinfo file
    gen_trackeval_files(dst_root, ['train','val','test'])