# Detector Documentation

The following detectors are supported out of the box. More/custom detectors can be easily added.

- Huggingface
- - e.g. [OWLv2](https://huggingface.co/docs/transformers/model_doc/owlv2), [Grounding Dino](https://huggingface.co/docs/transformers/model_doc/grounding-dino), [LLMDet](https://huggingface.co/fushh7/LLMDet)
- [mmDetection](https://github.com/open-mmlab/mmdetection)
- - e.g. [Grounding Dino](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/README.md)
- Already existing detections

## Huggingface Detectors

Using detectors from huggingface is straightforward and doesn't need any additional steps. The following arguments need to be set:

- `--DETECTOR.MODEL_SRC huggingface`
- `--DETECTOR.MODEL <model id>`

For example LLMDet large has the following huggingface model id: `iSEE-Laboratory/llmdet_large`. No additional changes are necessary.

## MMDetection

Using MMDetection models is rather complicated and not recommended. A new python environment needs to be set up with `mmengine, mmcv, mmdetection` according to the [official documentation](https://mmdetection.readthedocs.io/en/latest/get_started.html). Then model configs and checkpoints need to be downloaded and placed in the `mmcv` folder in this repository. Then the following arguments need to be set:

- `--DETECTOR.CHECKPOINT_PATH <path to the checkpoint file>`
- `--DETECTOR.CONFIG_PATH <path to the config file>`

**Warning:** A different environment is then used for detections and tracking. It is needed to precompute all detections, activate the standard environment and then start the tracking.

## Loading existing Detections

Existing detections can be loaded by setting the following arguments.

- `--DETECTOR.LOAD_DETS True`
- `--DETECTOR.MODEL <model name>`

The detections need to be arranged in the following way.

```
outputs/
├── <dataset name>/
│   ├── <dataset split>/
│   │   ├── <model name>
│   │   │   ├── seq1.pt/
│   │   │   ├── seq2.jpg
│   │   │   └── seq3.jpg
```

The detections are expected to be lists of dicts of tensors. Each list element contains the detections of the respective frame. The dict has the keys `boxes`, `scores` and `labels`. The `boxes` are in absolute pixel coordinates in tlwh-format and have the shape `[N,4]`. The `scores` are the confidence scores between 0 and 1 and are a tensor of shape `[N]`. The `labels` are integers and a tensor of shape `[N]`.
