import os
from pathlib import Path
from PIL import Image


def generate_seqinfo_ini(sequence_dir, img_folder='img1', frame_rate=30):
    img_dir = Path(sequence_dir) / img_folder
    if not img_dir.exists():
        print(f"Skipping {sequence_dir}: '{img_folder}' folder not found.")
        return

    # Find image files
    image_files = sorted([f for f in img_dir.iterdir()
                         if f.suffix.lower() in ['.jpg', '.png']])
    if not image_files:
        print(
            f"Skipping {sequence_dir}: no image files found in {img_folder}.")
        return

    # Get image size from the first file
    with Image.open(image_files[0]) as img:
        im_width, im_height = img.size

    seq_length = len(image_files)
    seq_name = Path(sequence_dir).name

    seqinfo_path = Path(sequence_dir) / "seqinfo.ini"
    with open(seqinfo_path, "w") as f:
        f.write("[Sequence]\n")
        f.write(f"name={seq_name}\n")
        f.write(f"imDir={img_folder}\n")
        f.write(f"frameRate={frame_rate}\n")
        f.write(f"seqLength={seq_length}\n")
        f.write(f"imWidth={im_width}\n")
        f.write(f"imHeight={im_height}\n")
        f.write("imExt=.jpg\n")  # Change this if using PNG or other formats

    print(f"Generated: {seqinfo_path}")
    return True


def create_seqmap_file(dataset_dir, sequence_names, split):
    seqmap_path = Path(dataset_dir) / f"{split}_seqmap.txt"
    with open(seqmap_path, "w+") as f:
        f.write("name\n")
        for name in sequence_names:
            f.write(f"{name}\n")
    print(f"Generated: {seqmap_path}")

def gen_trackeval_files(dataset_dir, splits):
    for split in splits:
        sequence_names = []
        parent_dir = os.path.join(dataset_dir, split)
        for subdir in os.listdir(parent_dir):
            full_path = os.path.join(parent_dir, subdir)
            if os.path.isdir(full_path):
                success = generate_seqinfo_ini(full_path)
                if success:
                    sequence_names.append(subdir)
        if sequence_names:
            create_seqmap_file(dataset_dir, sequence_names, split)
        else:
            print("No valid sequences found — no seqmap.txt created.")


