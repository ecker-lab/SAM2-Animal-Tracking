# Datasets Documentation

## Download

The following datasets are currently supported, they can be downloaded from the respective link:

- Animals
- - [ChimpAct](https://github.com/ShirleyMaxx/ChimpACT): Chimpanzees in the Leibzig Zoo.
- - [Bird Flock Tracking (BFT)](https://github.com/George-Zhuang/NetTrack): Different bird species in diverse Envionments.
- - [AnimalTrack](https://hengfan2010.github.io/projects/AnimalTrack/evaluation.html): Diverse selection of 10 common animal categories.
- - [GMOT-40-Animal)](https://github.com/Spritea/GMOT40): 4 different animal categories in crowded scenarios.
- - [PanAf500](https://obrookes.github.io/panaf.github.io/): Camer trap videos of chimpanzees in their natural environment.
- Persons
- - [DanceTrack](https://github.com/DanceTrack/DanceTrack): Dancers with unifrom appearance and diverse motion.
- - [SportsMot](https://github.com/MCG-NJU/SportsMOT): Athletes in diverse sport scenes.
- Vehicles
- - [UAVDT](https://sites.google.com/view/grli-uavdt/首页): Vehicles in complex scenes filmed with drones.
- - [BDD100k](https://bair.berkeley.edu/blog/2018/05/30/bdd/): Driving videos with multiple object classes.

## Processing

After downloading, the datasets need to be unpacked and assembled in the `data` folder. The goal is to bring them into the standard dancetrack format:

```
data/
├── dataset/
│   ├── train/
│   │   ├── seq1/
│   │   │   ├── img1/
│   │   │   │   ├── 00001.jpg
│   │   │   │   └── 00002.jpg
│   │   │   └── gt (optional)/
│   │   │       └── gt.txt
│   │   └── seq2/
│   └── test/
```

The `gt.txt` contains the annotations in the following format. The box coordinates are in tlwh format and in absolute pixel values. The frame IDs start with 1.

```
<frame_id>, <track_id>, <box_left>, <box_top>, <box_right>, <box_bottom>, 1, 1, 1
```

#### No Processing necessary

These datasets dont need to be processed. They are already in the correct format.

- DanceTrack
- SportsMOT

#### Processing according to Dataset

These datasets need to be processed according to their dataset website.

- ChimpAct

#### Custom Processing

These datasets need to be processed with the respective scripts in `./datasets`. The script convert the model into dancetrack format and add the auxiliary `seqmap` and `seqinfo` files for TrackEval.

- BFT
- AnimalTrack
- GMOT-40-Animal
- PanAf500

#### Special Case: BDD100k & UAVDT

These dataset have special characteristics. BDD100k is handled separately and can just be used as is. The results are also saved in a special format. UAVDT has some additional ignore regions, which also need to be supplied in addition to the ground truth.

## Adding Custom Dataset

A custom dataset can be added by putting it into the `data` folder in the dancetrack format and adding its paths and names to the dataset configs in run.py.

```
dataset_presets = {<dataset name>: {"DATASET": <dataset folder>, "IMG_DIR": <image dir>, "DETECTOR.TEXT_PROMPT": <detection text prompts>}
```

The dataset can then be used by specifying `--dataset <dataset name>` when calling `run.py`.
