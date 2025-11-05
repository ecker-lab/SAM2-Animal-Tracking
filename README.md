# Zero-Shot Multi-Animal Tracking in the Wild

## :memo: Overview

**TLDR.** We modify SAM2MOT for multi-animal tracking. Due to our adaptive detection threshold, no hyperparameter adaptations between different datasets are necessary. No adaptation, no finetuning, just throw it on your data (**fully zero-shot**).

![Overview](./assets/overview.png)

## :newspaper: News

- <span style="font-variant-numeric: tabular-nums;">**20.10.2025**</span>: A preliminary version of our paper is released on [arxiv](https://arxiv.org/abs/2511.02591).

## :rocket: Getting Started

**TLDR.** Follow the [installation instructions](./docs/INSTALL.md), add your dataset in [dancetrack format](./docs/DATASET.md) and run this command:

`python run.py --dataset <your dataset name>`

- See [INSTALL.md](./docs/INSTALL.md) for installation instructions.
- See [DATASET.md](./docs/DATASET.md) for dataset downloading and preprocessing.
- See [DETECTION.md](./docs/DETECTION.md) for information about supported detection models and loading detections.
- See [TRACKING.md](./docs/TRACKING.md) for how to run tracking on a dataset.

## :bar_chart: Results

| **Dataset**        | **HOTA↑** | **DetA↑** | **AssA↑** | **DetRe↑** | **LocA↑** | **MOTA↑** | **IDF1↑** | **IDSW↓** |
|:-------------------|:----------:|:----------:|:----------:|:------------:|:-----------:|:-----------:|:-----------:|:-----------:|
| ChimpAct        | 58.6 | 49.8 | 70.1 | 57.3 | 83.4 | 48.6 | 66.7 | 32   |
| BFT             | 74.8 | 72.2 | 77.7 | 80.5 | 87.8 | 81.8 | 88.4 | 51   |
| AnimalTrack     | 58.0 | 52.7 | 65.2 | 63.8 | 81.1 | 58.9 | 72.0 | 442  |
| GMOT-40-Animal  | 62.4 | 57.2 | 69.2 | 67.2 | 80.1 | 64.7 | 77.4 | 496  |

## :file_folder: Supported Datasets

The following datasets are supported out of the box. More/custom datasets can be easily added (See [DATASET.md](./docs/DATASET.md)).

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

## :hammer_and_wrench: Supported Detectors

The following detectors are supported out of the box. More/custom detectors can be easily added (See [DETECTION.md](./docs/DATASET.md)).

- Huggingface
- - e.g. [OWLv2](https://huggingface.co/docs/transformers/model_doc/owlv2), [Grounding Dino](https://huggingface.co/docs/transformers/model_doc/grounding-dino), [LLMDet](https://huggingface.co/fushh7/LLMDet)
- [mmDetection](https://github.com/open-mmlab/mmdetection)
- - e.g. [Grounding Dino](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/README.md)
- Already existing detections

## :handshake: Acknowledgements

This project is build upon [SAM2](https://github.com/facebookresearch/sam2), [SAM2MOT](https://github.com/TripleJoy/SAM2MOT) and [TrackEval](https://github.com/JonathonLuiten/TrackEval). We thank the authors for their amazing work.

## :books: Citation

If you think this project is helpful, please feel free to leave a :star: and cite our paper:

```
@misc{meier2025zeroshotmultianimaltrackingwild,
      title={Zero-Shot Multi-Animal Tracking in the Wild}, 
      author={Jan Frederik Meier and Timo Lüddecke},
      year={2025},
      eprint={2511.02591},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.02591}, 
}
```
