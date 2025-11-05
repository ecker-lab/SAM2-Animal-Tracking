import numpy as np
import torch
from torchvision.ops.boxes import box_area
import csv


def mask_to_xyxy(masks: np.ndarray) -> np.ndarray:
    """from supervision, converts a 3D `np.array` of 2D bool masks into a 2D `np.array` of bounding boxes.

    Args:
        masks (np.ndarray): A 3D `np.array` of shape `(N, W, H)`
            containing 2D bool masks

    Returns:
        np.ndarray: A 2D `np.array` of shape `(N, 4)` containing the bounding boxes
            `(x_min, y_min, x_max, y_max)` for each mask
    """
    n = masks.shape[0]
    xyxy = np.zeros((n, 4), dtype=int)

    for i, mask in enumerate(masks):
        rows, cols = np.where(mask)

        if len(rows) > 0 and len(cols) > 0:
            x_min, x_max = np.min(cols), np.max(cols)
            y_min, y_max = np.min(rows), np.max(rows)
            xyxy[i, :] = [x_min, y_min, x_max, y_max]

    return xyxy


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """ from DETR, Generalized IoU from https://giou.stanford.edu/

    Args:
        boxes1 (torch.Tensor): boxes in [x0, y0, x1, y1] format
        boxes2 (torch.Tensor): boxes in [x0, y0, x1, y1] format

    Returns:
        a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ from DETR, returns the box IoU and union

    Args:
        boxes1 (torch.Tensor): boxes in [x0, y0, x1, y1] format
        boxes2 (torch.Tensor): boxes in [x0, y0, x1, y1] format

    Returns:
        tuple[torch.Tensor, torch.Tensor]: intersection, union of both boxes
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def mask_iou(masks1: np.ndarray, masks2: np.ndarray) -> np.ndarray:
    """calculates iou of two sets of masks

    Args:
        masks1 (np.ndarray): first set of masks
        masks2 (np.ndarray): second set of masks

    Returns:
        np.ndarray: pairwise mask iou
    """
    intersection = np.sum(masks1 * masks2)
    union = np.sum(masks1 + masks2)

    return intersection / (union + 1e-7)


def is_inside_tlwh(pred:list, ignore:list) -> list:
    """removes box prediction if it is inside of an ignore area. Only needed for UAVDT dataset

    Args:
        pred (list): list of predictions in motchallenge format
        ignore (list): list of ignore regions in motchallenge format

    Returns:
        list: filtered predictions
    """
    px1, py1, pw, ph = map(float, pred[2:6])
    px2, py2 = px1 + pw, py1 + ph

    ix1, iy1, iw, ih = map(float, ignore[2:6])
    ix2, iy2 = ix1 + iw, iy1 + ih

    return (px1 >= ix1 and
            py1 >= iy1 and
            px2 <= ix2 and
            py2 <= iy2)


def preprocess_coco_targets(targets:list[dict], num_frames:int) -> list[dict]:
    """takes target in coco-format and brings it into expected format

    Args:
        targets (list[dict]): gt in coco-format
        num_frames (int): number of frames

    Returns:
        list[dict]: gt in dict per frame with keys "boxes" and "labels"
    """
    target = []
    for i in range(num_frames):
        if i < len(targets):
            boxes = torch.tensor([obj['bbox'] for obj in targets[i]])
            labels = torch.tensor([obj['category_id']
                                   for obj in targets[i]])
            target.append({'boxes': boxes, 'labels': labels})
        else:
            target.append({'boxes': torch.empty(
                0, 4), 'labels': torch.empty(0)})
    return target


def filter_predictions(pred_file:str, ignore_file:str, out_file:str):
    """removes box predictions which are inside of ignore areas. Only needed for UAVDT dataset.

    Args:
        pred_file (str): path to prediction file
        ignore_file (str): path to file with ignore areas
        out_file (str): output path
    """
    # load predictions
    with open(pred_file, "r") as f:
        preds = [row for row in csv.reader(f) if row]

    # load ignore areas
    with open(ignore_file, "r") as f:
        ignores = [row for row in csv.reader(f) if row]

    # group ignore boxes by frame
    ignores_by_frame = {}
    for ig in ignores:
        frame_id = int(ig[0])
        ignores_by_frame.setdefault(frame_id, []).append(ig)

    kept = []
    for pred in preds:
        frame_id = int(pred[0])
        frame_ignores = ignores_by_frame.get(frame_id, [])
        if not any(is_inside_tlwh(pred, ig) for ig in frame_ignores):
            kept.append(pred)

    # write filtered predictions
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(kept)