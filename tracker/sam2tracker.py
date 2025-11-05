import numpy as np
import torch
import os
import sys
from PIL import Image
from scipy.optimize import linear_sum_assignment
from collections import deque
import cv2
from tqdm import tqdm
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor

from utils.box_ops import mask_to_xyxy, generalized_box_iou, mask_iou
from detector import Detector


@torch.inference_mode()
def custom_propagate_in_video(
    self,
    inference_state,
    start_frame_idx=None,
    max_frame_num_to_track=None,
    reverse=False,
):
    """Modified version of the original sam 2 function. This one call the propagate_in_video_preflight function
    in each time step to initialize and reconstruct new tracks. SAM 2 license restrictions apply."""
    self.propagate_in_video_preflight(inference_state)

    obj_ids = inference_state["obj_ids"]
    num_frames = inference_state["num_frames"]
    batch_size = self._get_obj_num(inference_state)

    # set start index, end index, and processing order
    if start_frame_idx is None:
        # default: start from the earliest frame with input points
        start_frame_idx = min(
            t
            for obj_output_dict in inference_state["output_dict_per_obj"].values()
            for t in obj_output_dict["cond_frame_outputs"]
        )
    if max_frame_num_to_track is None:
        # default: track all the frames in the video
        max_frame_num_to_track = num_frames
    if reverse:
        end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
        if start_frame_idx > 0:
            processing_order = range(
                start_frame_idx, end_frame_idx - 1, -1)
        else:
            processing_order = []  # skip reverse tracking if starting from frame 0
    else:
        end_frame_idx = min(
            start_frame_idx + max_frame_num_to_track, num_frames - 1
        )
        processing_order = range(start_frame_idx, end_frame_idx + 1)

    for frame_idx in tqdm(processing_order, desc="propagate in video"):

        # added this block
        self.propagate_in_video_preflight(inference_state)
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)
        pred_masks_per_obj = [None] * batch_size

        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            # We skip those frames already in consolidated outputs (these are frames
            # that received input clicks or mask). Note that we cannot directly run
            # batched forward on them via `_run_single_frame_inference` because the
            # number of clicks on each object might be different.

            inference_state['curr_obj_id'] = inference_state['obj_idx_to_id'][obj_idx]

            if frame_idx in obj_output_dict["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = obj_output_dict[storage_key][frame_idx]
                device = inference_state["device"]
                pred_masks = current_out["pred_masks"].to(
                    device, non_blocking=True)
                if self.clear_non_cond_mem_around_input:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_obj_non_cond_mem_around_input(
                        inference_state, frame_idx, obj_idx
                    )
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=obj_output_dict,
                    frame_idx=frame_idx,
                    batch_size=1,  # run on the slice of a single object
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=True,
                )
                obj_output_dict[storage_key][frame_idx] = current_out

            inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                "reverse": reverse
            }
            pred_masks_per_obj[obj_idx] = pred_masks

        # Resize the output mask to the original video resolution (we directly use
        # the mask scores on GPU for output to avoid any CPU conversion in between)
        if len(pred_masks_per_obj) > 1:
            all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
        else:
            all_pred_masks = pred_masks_per_obj[0]
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, all_pred_masks
        )
        yield frame_idx, obj_ids, video_res_masks    


class SAM2Tracker(torch.nn.Module):
    def __init__(self, configs: dict, device: torch.device) -> None:
        """ initializes sam2tracker model

        Args:
            configs (dict): model configs
            device (torch.device): on which device the computations are performed (cuda or cpu)
        """
        super().__init__()
        # detection path
        self.load_dets = configs['DETECTOR']['LOAD_DETS']
        self.det_dir = os.path.join(configs['OUTPUT_DIR'], configs['DATASET'], configs['DATA_SPLIT'], configs['DETECTOR']['MODEL'])

        # init models
        SAM2VideoPredictor.propagate_in_video = custom_propagate_in_video
        self.sam2_predictor = self.init_sam2(configs['SAM2'], device)
        configs['DETECTOR']['TH_DET'] = configs['SAM2MOT']['TH_DET']
        self.det_model = Detector(configs['DETECTOR'])

        # object addition properties
        self.use_otsu = configs['SAM2MOT']['USE_OTSU']
        self.th_otsu = configs['SAM2MOT']['TH_OTSU']
        self.th_det = configs['SAM2MOT']['TH_DET']
        self.th_overlap = configs['SAM2MOT']['TH_OVERLAP']
        self.th_high_conf = configs['SAM2MOT']['TH_HIGH_CONF']
        self.known_ids = set()
        self.th_mask_empty = configs['SAM2MOT']['TH_MASK_EMPTY']

        # object removal and quality reconstruction properties
        self.th_iou_diff = configs['SAM2MOT']['TH_IOU_DIFF']
        self.th_reliable = configs['SAM2MOT']['TH_RELIABLE']
        self.th_pending = configs['SAM2MOT']['TH_PENDING']
        self.th_suspicious = configs['SAM2MOT']['TH_SUSPICIOUS']
        self.tol_frames = configs['SAM2MOT']['TOL_FRAMES']
        self.lost_count = {}

        # cross object interaction properties
        self.n_frames = configs['SAM2MOT']['N_FRAMES']
        self.th_miou = configs['SAM2MOT']['TH_MIOU']
        self.th_score_diff = configs['SAM2MOT']['TH_SCORE_DIFF']
        self.th_std_diff = configs['SAM2MOT']['TH_STD_DIFF']
        self.score_queue = deque(maxlen=self.n_frames)

        # mask nms
        self.mask_nms = configs['SAM2MOT']['MASK_NMS']
        self.th_nms_miou = configs['SAM2MOT']['TH_NMS_MIOU']

        # multiclass handling
        self.track_labels = {}

    def init_sam2(self, sam2_configs: dict, device: torch.device) -> SAM2VideoPredictor:
        """ initializes sam2

        Args:
            det_configs (dict): configs related to sam2
            device (torch.device): on which device the computations are performed (cuda or cpu)

        Returns:
            sam2 model
        """

        # full compile is not working when accessing internal states during propagation
        if sam2_configs['COMPILE_ENCODER']:
            extras = ['+compile_image_encoder=True']
        else:
            extras = []

        sam2_predictor = build_sam2_video_predictor(
            sam2_configs['CONFIG'], sam2_configs['CHECKPOINT'], hydra_overrides_extra=extras, device=device)
        return sam2_predictor

    def reset(self):
        """ resets states of sam2 and sam2mot after every sequence
        """
        self.lost_count = {}
        self.track_labels = {}
        self.known_ids = set()
        self.score_queue = deque(maxlen=self.n_frames)

    def object_addition(self, video_segment: dict, detections: dict) -> tuple[dict, tuple, np.ndarray]:
        """ adds new objects

        Args:
            video_segment (dict): masks from current frame
            detections (dict): detections from current frame

        Returns:
            tuple[dict, tuple, np.ndarray]: new_id-box pairs, hungarian matching results, ids
        """

        # preprocess video segment
        ids_list = list(video_segment.keys())
        # ids need to be unique, not every previously id is currently active
        self.known_ids.update(ids_list)
        ids = np.array(ids_list)
        masks = np.array(list(video_segment.values())).squeeze(1)
        sam_boxes = torch.from_numpy(
            mask_to_xyxy(masks)).to(dtype=torch.float32)

        # there is no valid mask for every id
        # some masks are false everywhere and we dont want those to be part of the hungarian matching
        nonzero_mask = sam_boxes.abs().sum(dim=1) != 0
        sam_boxes = sam_boxes[nonzero_mask, :]
        ids = ids[np.array(nonzero_mask)]

        # preprocess detections
        boxes = detections["boxes"]
        scores = detections["scores"].cpu()
        labels = detections["labels"].cpu()

        # filter "high confidence" detections
        # paper gives no further information except that the threshold is changed dynamically depending on the sequence
        cand_mask = scores >= self.th_high_conf
        high_conf_scores = scores[cand_mask]
        high_conf_boxes = boxes[cand_mask, :]
        high_conf_labels = labels[cand_mask]

        # calculate cost matrix
        cost_giou = - generalized_box_iou(high_conf_boxes, sam_boxes)
        C = cost_giou

        # hungarian matching between tracklets and detections
        # row_ind denotes the detection box and col_ind the tracklet box (row,col)
        # row_ind, col_ind = linear_sum_assignment(C)
        raw_indices = linear_sum_assignment(C)

        # only associate boxes when there is a minimum giou
        row_ind = []
        col_ind = []
        for i in range(len(raw_indices[0])):
            if C[raw_indices[0][i], raw_indices[1][i]] < - 0.2:
                row_ind.append(raw_indices[0][i])
                col_ind.append(raw_indices[1][i])
        indices = (np.array(row_ind), np.array(col_ind))

        # test candidate new objects
        cand_boxes = [b for i, b in enumerate(
            high_conf_boxes) if i not in indices[0]]
        max_id = max(self.known_ids)
        # only relevant when ablating sam2mot standard box-mask overlap
        if cand_boxes and self.th_overlap > 0:
            inv_mask_accum = np.sum(masks, 0) == 0
            overlap_ratios = []
            for box in cand_boxes:
                disc_box = torch.round(box).to(dtype=torch.int32)
                area = (disc_box[2] - disc_box[0]) * \
                    (disc_box[3] - disc_box[1])
                overlap = np.sum(
                    inv_mask_accum[disc_box[1]:disc_box[3], disc_box[0]:disc_box[2]])
                overlap_ratios.append(overlap / area)

                new_boxes = [box for box, p in zip(
                    cand_boxes, overlap_ratios) if p >= self.th_overlap]
                new_labels = [label for label, p in zip(
                    high_conf_labels, overlap_ratios) if p >= self.th_overlap]
                new_ids = [max_id + i + 1 for i in range(len(new_boxes))]
                new_objects = {id: (box, label) for id, box,
                           label in zip(new_ids, new_boxes, new_labels)}
        elif cand_boxes:
            new_ids = [max_id + i + 1 for i in range(len(cand_boxes))]
            new_objects = {id: (box, label) for id, box,
                           label in zip(new_ids, cand_boxes, high_conf_labels)}
        else:
            new_objects = {}

        return new_objects, indices, ids

    def object_removal(self, seg_scores: dict) -> set:
        """performs object_removal according to sam2mot paper

        Args:
            seg_scores (dict): mask scores from current objects

        Returns:
            list: object ids for removal
        """
        remove_ids = []
        for id, score in seg_scores.items():
            if score <= self.th_suspicious:
                self.lost_count[id] = self.lost_count.get(id, 0) + 1
                if self.lost_count[id] >= self.tol_frames:
                    remove_ids.append(id)
            elif score > self.th_pending:
                self.lost_count[id] = 0

        return set(remove_ids)

    def quality_reconstruction(self, detections: dict, indices: tuple, ids: np.ndarray, seg_scores: dict) -> dict:
        """ performs quality reconstruction according to sam2mot paper

        Args:
            detections (dict): detections from current frame
            indices (tuple): results from hungarian matching (performed in add_object)
            ids (np.ndarray): array of current ids
            seg_scores (dict): mask scores from current objects

        Returns:
            dict: id-box pairs for objects, which should be reconstructed
        """

        if len(indices[0]) == 0:
            return {}

        # preprocess detections
        boxes = detections["boxes"].cpu()
        scores = detections["scores"].cpu()
        labels = detections["labels"].cpu()

        # get high confidence detections
        high_conf_mask = scores >= self.th_high_conf
        high_conf_boxes = boxes[high_conf_mask, :]
        high_conf_labels = labels[high_conf_mask]

        # get matched high confidence detections
        matched_ids = ids[indices[1]]
        recon_cand_boxes = high_conf_boxes[indices[0], :]
        recon_labels = high_conf_labels[indices[0]]

        # check if corresponding segmented objects are uncertain
        recon_mask = [True if self.th_pending <= seg_scores[id].item(
        ) <= self.th_reliable else False for id in matched_ids]
        recon_boxes = recon_cand_boxes[recon_mask, :]
        recon_ids = matched_ids[recon_mask]

        # multiclass handling
        box_labels = recon_labels[recon_mask]
        track_labels = np.array([self.track_labels[rid] for rid in recon_ids])
        label_mask = box_labels == track_labels
        recon_ids = [recon_ids[i]
                     for i in range(len(recon_ids)) if label_mask[i]]
        recon_boxes = [recon_boxes[i]
                       for i in range(len(recon_boxes)) if label_mask[i]]

        recon_objects = {id: box for id, box in zip(recon_ids, recon_boxes)}

        return recon_objects

    def cross_object_interaction(self, score_logits: dict, video_segment: dict) -> set:
        """ performs cross-object-interaction according to sam2mot paper

        Args:
            score_logits (dict): scores of segmented objects from the past n frames
            video_segment (dict): masks of segmented objects in the current frame

        Returns:
            set: ids on which to perform cross-object-interaction
        """
        self.score_queue.append(score_logits)

        ids = list(video_segment.keys())
        masks = list(video_segment.values())

        cands = []
        remove_mem = []

        # detect all mIoUs which are above a threshold
        # those are the possible cross object interaction candidates
        for i in range(len(ids)):
            for j in range(len(ids)):
                if i <= j:
                    continue
                m_iou = mask_iou(masks[i], masks[j])
                if m_iou > self.th_miou:
                    cands.append((i, j))

        # check if the candidates have a score or std difference above a certain threshold
        for (cand_i, cand_j) in cands:
            
            scores_i = torch.tensor([scores.get(
                ids[cand_i]) for scores in self.score_queue if scores.get(ids[cand_i]) is not None])
            scores_j = torch.tensor([scores.get(
                ids[cand_j]) for scores in self.score_queue if scores.get(ids[cand_j]) is not None])
            std_i = torch.std(scores_i)
            std_j = torch.std(scores_j)

            if scores_i[0] - scores_j[0] > self.th_score_diff:
                remove_mem.append(ids[cand_j])
            elif scores_j[0] - scores_i[0] > self.th_score_diff:
                remove_mem.append(ids[cand_i])
            elif std_i - std_j > self.th_std_diff:
                remove_mem.append(ids[cand_i])
            elif std_j - std_i > self.th_std_diff:
                remove_mem.append(ids[cand_j])

        return set(remove_mem)

    def mask_non_max_supp(self, mask_logits: torch.Tensor, ids: list, inference_state: dict) -> list:
        """ perform nms on masks, removes younger mask

        Args:
            mask_logits (torch.Tensor): segmentation masks in tensor format [N,1,H,W]
            ids (list): corresponding object ids
            inference_state (dict): information about current tracks from sam2

        Returns:
            list: which ids to remove
        """
        remove_nms_ids = []

        for i, id1 in enumerate(ids):
            mask1 = (mask_logits[i, 0, :, :] > 0.0).cpu().numpy()
            for j, id2 in enumerate(ids):
                if id1 >= id2:
                    continue
                mask2 = (mask_logits[j, 0, :, :] > 0.0).cpu().numpy()
                miou = mask_iou(mask1, mask2)
                if miou > self.th_nms_miou:

                    # remove younger object (appeared first in a later frame)
                    age_id1 = min(list(
                        inference_state['output_dict_per_obj'][inference_state['obj_id_to_idx'][id1]]['cond_frame_outputs'].keys()))
                    age_id2 = min(list(
                        inference_state['output_dict_per_obj'][inference_state['obj_id_to_idx'][id2]]['cond_frame_outputs'].keys()))
                    remove_nms_ids.append(id1 if age_id1 >= age_id2 else id2)

        return remove_nms_ids

    def init_sequence(self, video_dir: str) -> tuple[dict, dict, list]:
        """ itializes sam2mot on the first frame of the sequence, eg. load frames, detect objects, prompt sam2, ...

        Args:
            video_dir (str): directory, where the image frames are located

        Returns:
            tuple[dict, dict, list]: detections in the first frame, sam2 inference state and frame names
        """

        # load saved detections if there are any
        if self.load_dets:
            seq = os.path.basename(os.path.dirname(video_dir))
            self.det_model.load_detections(os.path.join(self.det_dir, seq + '.pt'))

            # calculate adaptive threshold from detection score distribution
            if self.use_otsu:
                scores = np.array(
                    [d for det in self.det_model.detections for d in det['scores']])
                scores_scaled = (255 * scores).astype(np.uint8).reshape(-1, 1)
                otsu_threshold, _ = cv2.threshold(
                    scores_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
                self.th_det = otsu_threshold / 255
                self.th_high_conf = otsu_threshold / 255 + self.th_otsu
                self.det_model.th_det = self.th_det
        else:
            self.detections_seq = None

        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]

        def extract_number(p):
            # Take the base filename (without extension)
            base = os.path.splitext(p)[0]  # e.g. "frame_10"
            # Find the last sequence of digits
            match = re.search(r'\d+', base)
            return int(match.group())

        frame_names.sort(key=extract_number)

        # initialize sam2
        inference_state = self.sam2_predictor.init_state(video_path=video_dir)
        ann_frame_idx = 0

        # get initial detections
        img_path = os.path.join(video_dir, frame_names[ann_frame_idx])
        image = Image.open(img_path)
        detections = self.det_model([image], ann_frame_idx)[0]
        input_boxes = detections['boxes']

        if len(detections['boxes']) == 0:
            detections = {'scores': torch.tensor([0]), 'boxes': torch.tensor(
                [0, 0, 0, 0]).reshape((1, 4)), 'labels': torch.tensor([-1])}
            input_boxes = detections['boxes']

        # prompt sam2 with detected boxes
        for object_id, box in enumerate(input_boxes):
            _, _, _ = self.sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id if torch.any(box != 0) else -1,
                box=box.float().numpy(),
            )
            if torch.any(box != 0):
                self.track_labels[object_id] = int(
                    detections['labels'][object_id])

        return detections, inference_state, frame_names

    def forward_sequence(self, video_dir: str) -> tuple[dict, dict, list]:
        """ iterates with the sam2tracker over a given sequence

        Args:
            video_dir (str): directory, where the image frames are located

        Returns:
            tuple[dict, dict, list]: frame-wise segmentation masks, detections and frame names
        """

        # detect objects and prompt sam2 on initial frame
        detections, inference_state, frame_names = self.init_sequence(
            video_dir)

        # video_segments contains the per-frame segmentation results, video_dets per-frame detections
        video_segments = {}
        video_dets = {}

        # run propagation throughout the video and collect the results in a dict
        for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_predictor.propagate_in_video(inference_state):
            inference_state['frame_idx'] = out_frame_idx

            # all modifications are only active from the second frame on
            if out_frame_idx > 0:

                # extract score logits from inference state
                score_logits = {}
                for out_obj_id in out_obj_ids:
                    try:
                        scores = inference_state["output_dict_per_obj"][inference_state["obj_id_to_idx"].get(
                            out_obj_id, None)]["non_cond_frame_outputs"][out_frame_idx]['object_score_logits']
                    except:
                        scores = inference_state["output_dict_per_obj"][inference_state["obj_id_to_idx"].get(
                            out_obj_id, None)]["cond_frame_outputs"][out_frame_idx]['object_score_logits']
                    score_logits[out_obj_id] = scores

                # extract sam2 predictions
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

                # detect objects in each frame
                img_path = os.path.join(video_dir, frame_names[out_frame_idx])
                image = Image.open(img_path)
                detections = self.det_model([image], out_frame_idx)[0]

                # sam2mot functions are called here
                new_objects, indices, ids = self.object_addition(
                    video_segments[out_frame_idx], detections)
                recon_objects = self.quality_reconstruction(
                    detections, indices, ids, score_logits)
                remove_ids = self.object_removal(score_logits)

                # mask nms function
                if self.mask_nms:
                    remove_nms_ids = self.mask_non_max_supp(
                        out_mask_logits, out_obj_ids, inference_state)
                    remove_ids.update(remove_nms_ids)

                remove_mem_ids = self.cross_object_interaction(
                    score_logits, video_segments[out_frame_idx])

                # object addition
                # adds new object by prompting with detection bbox
                label_mod = 0
                for new_id, (new_box, new_label) in new_objects.items():
                    # get mask for each new object candidate
                    _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_points_or_box(
                        inference_state,
                        out_frame_idx,
                        new_id + label_mod,
                        box=new_box.float().numpy(),
                    )

                    # check if candidates fulfill requirements
                    new_obj_mask = (out_mask_logits[-1][0] > 0).cpu().numpy()
                    for mask in out_mask_logits[:-1]:
                        mask = (mask[0] > 0).cpu().numpy()
                        miou = np.sum(new_obj_mask * mask) / \
                            np.sum(new_obj_mask)
                        
                        # if not, candidates are removed from tracked objects
                        if miou > self.th_mask_empty:
                            self.sam2_predictor.remove_object(
                                inference_state, new_id + label_mod, strict=True, need_output=False)
                            label_mod -= 1
                            out_mask_logits = out_mask_logits[:-1, ...]
                            out_obj_ids = out_obj_ids[:-1]
                            break

                    # print(f'Added object with id {new_id}')
                    self.track_labels[new_id] = new_label

                # cross object interaction
                # removes occluded object from memory attention
                for id in remove_mem_ids:
                    # print(f'Removed occluded object with id {id} from current frame')
                    if out_frame_idx in inference_state['output_dict_per_obj'][inference_state['obj_id_to_idx']
                                                                               [id]]["non_cond_frame_outputs"]:
                        del inference_state['output_dict_per_obj'][inference_state['obj_id_to_idx']
                                                                   [id]]["non_cond_frame_outputs"][out_frame_idx]

                # object reconstruction
                # reprompt existing object with well aligned detection bbox
                for recon_id, recon_box in recon_objects.items():
                    if recon_id in remove_mem_ids:
                        continue
                    
                    # get boxes from masks and calculate all IoUs
                    iou_box = []
                    for id, mask in video_segments[out_frame_idx].items():
                        mask_box = mask_to_xyxy(mask)
                        iou_box.append(
                            generalized_box_iou(recon_box.unsqueeze(0), torch.tensor(mask_box)))

                    # dont need to check difference to second best detection, if there is only one
                    if len(iou_box) == 1:
                        continue

                    # check difference between best and second-best match
                    first_to_sec = max(iou_box) - sorted(iou_box)[-2]
                    if first_to_sec < self.th_iou_diff:
                        continue

                    # print(f'Reconstructed object with id {recon_id}')

                    # reconstruction is implemented by removing the object and initializing it again with the new box
                    self.sam2_predictor.remove_object(
                        inference_state, recon_id, strict=True, need_output=False)
                    _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_points_or_box(
                        inference_state,
                        out_frame_idx,
                        recon_id.item(),
                        box=recon_box.float().numpy(),
                    )

                # object removal
                for id in remove_ids:
                    # do not remove object if it is the last one (sam2 limitation), will not be in output
                    if len(out_obj_ids) - len(remove_ids) > 0:
                        if id not in out_obj_ids:
                            continue
                        # print(f'Removed object with id {id}')
                        self.sam2_predictor.remove_object(
                            inference_state, id, strict=True, need_output=False)

            # save updated segmentation masks and detection
            video_dets[out_frame_idx] = detections
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        track_labels = self.track_labels.copy()

        # reset states before tracking next sequence
        inference_state.clear()
        self.reset()

        return video_segments, video_dets, frame_names, track_labels
