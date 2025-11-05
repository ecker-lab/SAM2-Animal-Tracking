import argparse
import subprocess
import os
import json
from pathlib import Path

from tracker.utils.misc import str2bool, nest_config, PresetAction
from tracker.utils.box_ops import filter_predictions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)


def parse_args():
    # define multi-config presets (datasets)
    dataset_presets = {"dancetrack": {"DATASET": "dancetrack", "IMG_DIR": "img1", "DETECTOR.TEXT_PROMPT": [["person"]]},
                       "sportsmot": {"DATASET": "sportsmot", "IMG_DIR": "img1", "DETECTOR.TEXT_PROMPT": [["athlete"]]},
                       "chimpact": {"DATASET": "ChimpACT_processed", "IMG_DIR": "", "DETECTOR.TEXT_PROMPT": [["ape"]]},
                       "bft": {"DATASET": "BFT_dancetrack", "IMG_DIR": "img1",
                               "DETECTOR.TEXT_PROMPT": [["bird"]]},
                       "panaf500": {"DATASET": "panaf500_tracking", "IMG_DIR": "img1", "DETECTOR.TEXT_PROMPT": [["ape"]]},
                       "bdd100k": {"DATASET": "bdd100k", "IMG_DIR": "", "DETECTOR.TEXT_PROMPT": [["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]]},
                       "uavdt": {"DATASET": "UAV-benchmark-M", "IMG_DIR": "", "DETECTOR.TEXT_PROMPT": [["car"]]},
                       "gmot40": {"DATASET": "gmot-40_dancetrack", "IMG_DIR": "img1", "DETECTOR.TEXT_PROMPT": [["airplane", "ball", "balloon", "bird", "car", "fish", "insect", "person", "cow", "sheep", "goat", "wolf"]]},
                       "gmot40-animal": {"DATASET": "gmot-40-animal", "IMG_DIR": "img1", "DETECTOR.TEXT_PROMPT": [["bird", "fish", "insect", "sheep", "goat", "cow", "wolf"]]},
                       "animaltrack": {"DATASET": "AnimalTrack_dancetrack", "IMG_DIR": "img1", "DETECTOR.TEXT_PROMPT": [["chicken", "deer", "dolphin", "duck", "goose", "horse", "penguin", "pig", "rabbit", "zebra"]]}
                       }

    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument(
        "--dataset",
        action=PresetAction,
        presets=dataset_presets,
        choices=list(dataset_presets.keys()),
        help="Choose speed preset affecting lr, batch_size, num_epochs"
    )
    parser.add_argument("--DATA_SPLIT", type=str, default="val")
    parser.add_argument("--DATA_ROOT", type=str,
                        default="data")

    # task args
    parser.add_argument('--TRACKING', type=str2bool, default=True)
    parser.add_argument('--DETECTION', type=str2bool, default=True)
    parser.add_argument('--EVALUATION', type=str2bool, default=True)

    # output args
    parser.add_argument("--TEST_NAME", type=str, default="test")
    parser.add_argument("--OUTPUT_DIR", type=str, default="outputs")
    parser.add_argument("--SAVE_IMGS", type=str2bool, default=False)
    parser.add_argument("--SAVE_VIDS", type=str2bool, default=False)
    parser.add_argument("--SAVE_PREDS", type=str2bool, default=True)
    parser.add_argument("--SAVE_DETS", type=str2bool, default=True)

    # SAM2 settings
    parser.add_argument("--SAM2.CHECKPOINT", type=str,
                        default="sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--SAM2.CONFIG", type=str,
                        default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--SAM2.COMPILE_ENCODER", type=str2bool, default=True)

    # detector settings
    parser.add_argument("--DETECTOR.LOAD_DETS", type=str2bool,
                        default=False)
    parser.add_argument("--DETECTOR.BATCH_SIZE", type=int, default=1)
    parser.add_argument("--DETECTOR.MODEL_SRC",
                        type=str, default="huggingface")
    parser.add_argument("--DETECTOR.MODEL", type=str,
                        default="iSEE-Laboratory/llmdet_large")
    parser.add_argument("--DETECTOR.DEVICE", type=str,
                        default='cuda')
    parser.add_argument("--DETECTOR.COMPILE", type=str2bool,
                        default=False)
    parser.add_argument("--DETECTOR.CHECKPOINT_PATH", type=str,
                        default='mmcv/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth')
    parser.add_argument("--DETECTOR.CONFIG_PATH", type=str,
                        default='mmcv/configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg.py')
    parser.add_argument("--DETECTOR.USE_NMS", type=str2bool,
                        default=False)
    parser.add_argument("--DETECTOR.TH_NMS", type=float,
                        default=0.95)
    parser.add_argument("--DETECTOR.TH_DET", type=float,
                        default=0.1)
    parser.add_argument("--DETECTOR.OUT_FORMAT", type=str,
                        default='xywh')

    # SAM2MOT settings
    # object addition settings
    parser.add_argument("--SAM2MOT.USE_OTSU", type=str2bool, default=True)
    parser.add_argument("--SAM2MOT.TH_OTSU", type=float, default=0.1)
    parser.add_argument("--SAM2MOT.TH_DET", type=float, default=0.4)
    parser.add_argument("--SAM2MOT.TH_OVERLAP", type=float, default=-0.5) # legacy parameter for ablation
    parser.add_argument("--SAM2MOT.TH_HIGH_CONF", type=float, default=0.5)
    parser.add_argument("--SAM2MOT.TH_MIN_GIOU", type=float, default=0.2)
    parser.add_argument("--SAM2MOT.TH_MASK_EMPTY", type=float, default=0.4)

    # matching costs for hungarian matching (currently only using giou based matching)
    parser.add_argument("--SAM2MOT.COST_GIOU", type=int, default=1)

    # object removal and quality reconstruction settings
    parser.add_argument("--SAM2MOT.TH_IOU_DIFF", type=float, default=0.3)
    parser.add_argument("--SAM2MOT.TH_RELIABLE", type=float, default=8)
    parser.add_argument("--SAM2MOT.TH_PENDING", type=float, default=6)
    parser.add_argument("--SAM2MOT.TH_SUSPICIOUS", type=float, default=2)
    parser.add_argument("--SAM2MOT.TOL_FRAMES", type=int, default=25)

    # cross object interaction properties
    parser.add_argument("--SAM2MOT.N_FRAMES", type=int, default=10)
    parser.add_argument("--SAM2MOT.TH_MIOU", type=float, default=0.8)
    parser.add_argument("--SAM2MOT.TH_SCORE_DIFF", type=float, default=2)
    parser.add_argument("--SAM2MOT.TH_STD_DIFF", type=float, default=0.2)

    # masn nonmaximum suppression
    parser.add_argument("--SAM2MOT.MASK_NMS", type=str2bool, default=True)
    parser.add_argument("--SAM2MOT.TH_NMS_MIOU", type=float, default=0.95)

    args = parser.parse_args()
    config = nest_config(args)


    # chimpact original structure has an additional dir compared to standard dancetrack
    if config["DATASET"] == 'ChimpACT_processed':
        config["DATA_SPLIT"] = config["DATA_SPLIT"] + '/images'
    # same for bdd100k
    elif config["DATASET"] == 'bdd100k':
        config["DATA_SPLIT"] = 'images/' + 'track/' + config["DATA_SPLIT"]
    else:
        pass

    print(config)
    return config


def run_evaluation(config):
    
    dataset_dir = os.path.join(
        config['DATA_ROOT'], config['DATASET'], config['DATA_SPLIT'])
    output_dir = os.path.join(
        config['OUTPUT_DIR'], config['DATASET'], config['DATA_SPLIT'], config['TEST_NAME'])

    if config['DATASET'] == 'UAV-benchmark-M':

        ignore_dir = 'data/UAV-benchmark-MOTD_v1.0/GT'

        folder_a = Path(output_dir)
        folder_b = Path(ignore_dir)
        suffix = "_gt_ignore"   # the fixed suffix before extension

        for file_a in folder_a.iterdir():
            if file_a.is_file() and 'pedestrian' not in file_a.name:
                stem, ext = os.path.splitext(
                    file_a.name)  # split name + extension
                # construct matching filename
                file_b = folder_b / f"{stem}{suffix}{ext}"
                filter_predictions(file_a, file_b, file_a)

    args = {
        "--SPLIT_TO_EVAL": config['DATA_SPLIT'],
        "--METRICS": ["HOTA", "CLEAR", "Identity"],
        "--GT_FOLDER": dataset_dir,
        "--SEQMAP_FILE": os.path.join(config['DATA_ROOT'], config['DATASET'], f"{config['DATA_SPLIT']}_seqmap.txt"),
        "--SKIP_SPLIT_FOL": "True",
        "--TRACKERS_TO_EVAL": "",
        "--TRACKER_SUB_FOLDER": "",
        "--USE_PARALLEL": "True",
        "--NUM_PARALLEL_CORES": "8",
        "--PLOT_CURVES": "False",
        "--TRACKERS_FOLDER": output_dir,
        "--DO_PREPROC": "False",
    }
    cmd = ["python", "TrackEval/scripts/run_mot_challenge.py"]
    for k, v in args.items():
        cmd.append(k)
        if isinstance(v, list):
            cmd += v
        else:
            cmd.append(v)

    _ = subprocess.run(cmd,)


def run_tracking(config):
    # setup paths
    dataset_dir = os.path.join(
        config['DATA_ROOT'], config['DATASET'], config['DATA_SPLIT'])

    # iterate over all sequences in given dataset split
    sequences = os.listdir(dataset_dir)
    sequences.sort()
    for seq in sequences[:]:

        print('-----------------------')
        print(f'Start tracking for sequence "{seq}"')
        print('-----------------------')

        video_dir = os.path.join(dataset_dir, seq, config['IMG_DIR'])
        subprocess.run(["python", "tracker/track.py",
                       video_dir, seq, json.dumps(config)])


if __name__ == '__main__':
    config = parse_args()

    if config["DETECTION"]:
        subprocess.run(["python", "tracker/detect.py", json.dumps(config)])
        config["DETECTOR"]["LOAD_DETS"] = True

    if config["TRACKING"]:
        run_tracking(config)

    if config["EVALUATION"]:
        run_evaluation(config)
