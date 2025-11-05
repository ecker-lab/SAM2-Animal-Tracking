import torch.nn as nn
import torch
import numpy as np
from torchvision.ops import nms, box_convert
from torch import Tensor

from utils.misc import move_to, mask_dict, process_text_labels

class Detector(nn.Module):
    """unifies all detection models used (currently huggingface, mmdetection). Also loads existing detections."""

    def __init__(self, config: dict, out_box_format: str='xyxy') -> None:
        """initializes detection model

        Args:
            config (dict): config dict containing hyperparams and paths
            out_box_format (str, optional): output format of the bounding box. Defaults to 'xyxy'.
        """
        super().__init__()
        
        self.model_src = config['MODEL_SRC']
        self.model_name = config['MODEL']
        self.load_dets = config['LOAD_DETS']

        self.checkpoint_path = config['CHECKPOINT_PATH']
        self.config_path = config['CONFIG_PATH']
        self.device = config['DEVICE']
        self.compile_model = config['COMPILE']

        self.th_det = config['TH_DET']
        self.use_nms = config['USE_NMS']
        self.th_nms = config['TH_NMS']
        self.text_prompt = config['TEXT_PROMPT']

        self.out_box_format = out_box_format
        self.box_format = 'xyxy'
        self.det_model = None
        self.det_processor = None
        self.detections = None

        # check if detector is needed
        if self.load_dets:
            self.box_format = 'xywh'
        else:
            # models from huggingface
            if self.model_src in ['huggingface']:
                from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
                try:
                    self.det_processor = AutoProcessor.from_pretrained(self.model_name)
                    self.det_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_name).to(self.device)
                except:
                    raise ValueError(f"detection model '{self.model_name}' is currently not supported in huggingface.")
                
                if self.compile_model:
                    try:
                        det_model = torch.compile(
                            det_model, mode="max-autotune", fullgraph=True)
                    except:
                        print(
                            f"WARNING: could not torch.compile '{config['MODEL']}'")

            # models from mmdetections
            elif self.model_src in 'mmdetection':
                from mmdet.apis import DetInferencer
                try:
                    self.det_model = DetInferencer(model=self.config_path, weights=self.checkpoint_path,
                                        device=self.device,
                                        show_progress=False)
                except:
                    raise ValueError(f"detection model can't be initialized from this checkpoint '{self.checkpoint_path}' and config '{self.config_path}'.")
            
            # add own model by adding it here and in forward()
            else:
                raise ValueError(f"detection model source '{self.model_src}' is currently not supported")

    def forward(self, images: list[Tensor], frame_idx:int=None) -> dict:
        """detects objects/ loads detections from given images/indices

        Args:
            images (list[Tensor]): images
            frame_idx (int, optional): loads detection corresponding to this frame_id. Defaults to None.

        Returns:
            dict: detections {"boxes","labels","scores"}
        """
        bs = len(images)
        target_sizes = torch.tensor([images[i].size[::-1] for i in range(bs)])

        if self.load_dets:
            assert self.detections is not None
            detections = []
            for i in range(bs):
                if frame_idx + i >= len(self.detections):
                    break
                detections.append(self.detections[frame_idx + i])
        else:
            if self.model_src == 'huggingface':
                inputs = self.det_processor(text=self.text_prompt * bs, images=images, return_tensors="pt").to(
                        self.det_model.device)

                with torch.no_grad():
                    outputs_raw = self.det_model(**inputs)

                detections = self.det_processor.post_process_grounded_object_detection(
                        outputs=outputs_raw, target_sizes=target_sizes, threshold=self.th_det)
            elif self.model_src == 'mmdetection':
                # mmdetection expects images in bgr format instead of rgb
                images_bgr = []
                for img in images:
                    img_bgr = np.array(img)[..., ::-1]
                    images_bgr.append(img_bgr)

                detections = self.det_model(images_bgr,
                                texts=self.text_prompt)['predictions']
                # some detectors output boxes with the key "bboxes" instead of "boxes"
                detections = [
                    {('boxes' if k == 'bboxes' else k): torch.tensor(v)
                        for k, v in det.items()}
                    for det in detections
                ]
            else:
                pass
        # process detections (filter, reformat)
        detections = self.postprocess(detections, out_device='cpu')
        if len(detections) == 0:
            detections = [{'boxes': torch.empty([0,4]), 'labels': torch.empty([0]), 'scores':torch.empty([0])}]
        return detections

    def postprocess(self, detections:list[dict], out_device:str) -> list[dict]:
        """reformats and filters detections

        Args:
            detections (list[dict]): detections per frame
            out_device (str): 'cpu' or 'cuda'

        Returns:
            list[dict]: filtered detections per frame
        """

        if self.box_format not in ['xyxy', 'xywh']:
            raise ValueError(
                f'Unsupported input box format: {self.box_format}')
        
        # some detectors output the labels as strings, this converts them to ints
        detections = process_text_labels(detections, self.text_prompt[0])
        dets = move_to(detections, torch.device(out_device))
        
        for i in range(len(dets)):
            # threshold detections with low score 
            th_mask = dets[i]['scores'] >= self.th_det
            dets[i]['boxes'] = dets[i]['boxes'][th_mask]
            dets[i]['labels'] = dets[i]['labels'][th_mask]
            dets[i]['scores'] = dets[i]['scores'][th_mask]

            if dets[i]['labels'].numel() == 0:
                continue
            
            # apply nonmaximum suppression
            if self.use_nms:
                nms_idx = nms(
                    dets[i]['boxes'] if self.box_format == 'xyxy' else
                    box_convert(dets[i]['boxes'], 'xywh', 'xyxy'),
                    dets[i]['scores'],
                    iou_threshold=self.th_nms)
                nms_mask = torch.zeros_like(dets[i]['scores']) > 1
                nms_mask[nms_idx] = True
                dets[i] = mask_dict(dets[i], nms_mask)

            # convert box format
            if self.box_format != self.out_box_format:
                if self.out_box_format == 'xywh':
                    dets[i]['boxes'] = box_convert(dets[i]['boxes'], 'xyxy', 'xywh')
                elif self.out_box_format == 'xyxy':
                    dets[i]['boxes'] = box_convert(dets[i]['boxes'], 'xywh', 'xyxy')
                else:
                    raise ValueError(
                        f'Unsupported output box format: {self.out_box_format}')

        return dets
    
    def load_detections(self, det_path:str) -> None:
        """loads detections for whole sequence

        Args:
            det_path (str): path to saved detections
        """
        try:
            self.detections = torch.load(det_path)
        except:
            raise ValueError(f"Can't load detections from: {det_path}")