# Tracking

## Overview

The main function in this repository is `run.py`. It contains the argument parser and calls the respective subfunctions for detection, tracking and evaluation. The respective subtasks can be activated or deactivated by adding the following arguments when calling `run.py`.

- Detection: `--DETECTION True/False`
- Tracking: `--TRACKING True/False`
- Evaluation: `--EVALUATION True/False`

When setting `--DETECTION True`, the chosen detection model detects all object in the specified dataset and saves the detection. `--TRACKING True` then uses these detections if available and uses SAM2 and the heuristics on top to track the detected objects. `--EVALUATION True` then calls the TrackEval package to evaluate the tracks. 

**Warning:** When `--DETECTION False` and there are no saved detections, the model also detects during tracking. This means that it cant use the adaptive detection thresholds, since those require the detections to be available in the beginning.

## Important Arguments

The following arguments are useful for using the model. The have a default value specified in `run.py`, which can be easily changed

- `--DETECTION`: Detect all objects before tracking (default: True).
- `--TRACKING`: Track objects (default: True).
- `--EVALUATION`: Evaluate tracking (requires ground-truths) (default: True).

- `--TEST_NAME`: Name of the run (default: test).
- `--SAVE_IMGS`: Saves images with visualized predictions (default: False).
- `--SAVE_VIDS`: Converts saved images with predictions into video (default: False).

- `--LOAD_DETS`: Load existing detections (useful when there are ones or if `--DETECTION True` was used once before on the same dataset) (default: False).

- `--DATA_SPLIT`: Which datasplit to use (train/val/test) (default: val).
- `--DATA_ROOT`: Path to the parent directory of all datasets (default: data).
- `--dataset`: Which dataset to use.

## Example run

We want to track all objects on the ChimpAct test split, visualize the output predictions and evaluate the performance.

`python tracker/run.py --dataset chimpact --DATA_SPLIT test --SAVE_IMGS True --DETECTION True`