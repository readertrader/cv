# Taken from https://github.com/facebookresearch/detectron2
import math
import torch
from utils import calc_iou, enclosing_box_length, calc_iou_array, to_corners, DEVICE
import numpy as np


def modulated_loss(pred, label):
    x1, x2 = pred[:, 0], label[:, 0]
    y1, y2 = pred[:, 1], label[:, 1]
    yaw1, yaw2 = pred[:, 2], label[:, 2]
    w1, w2 = pred[:, 3], label[:, 3]
    h1, h2 = pred[:, 4], label[:, 4]
    lcp = torch.abs(x1 - x2) + torch.abs(y1 - y2)
    lmr = torch.min(
        lcp + torch.abs(w1 - w2) + torch.abs(h1 - h2) + torch.abs(yaw1 - yaw2),
        lcp
        + torch.abs(w1 - h2)
        + torch.abs(h1 - w2)
        + torch.abs(torch.pi/2 - torch.abs(yaw1 - yaw2)),
    )

    return lmr


def ciou_loss_batch(pred, gt, reduction='none'):
    losses = []
    ious = []
    for i in range(len(gt)):
        loss, iou = ciou_loss(pred[i], gt[i])
        losses.append(loss)
        ious.append(iou)
    losses = np.array(losses)
    ious = np.array(ious)
    if reduction == "mean":
        losses = losses.mean()
        ious = ious.mean()
    return losses, ious

def diou_loss_batch(pred, gt, reduction='none', yaw=False):
    losses = []
    ious = []
    for i in range(len(gt)):
        loss, iou = diou_loss(pred[i], gt[i], yaw)
        losses.append(loss)
        ious.append(iou)
    losses = np.array(losses)
    ious = np.array(ious)
    if reduction == "mean":
        losses = losses.mean()
        ious = ious.mean()
    return losses, ious

def ciou_loss(
    preds,
    labels,
    reduction='mean',
    eps: float = 1e-7,
):
    dloss, iou = diou_loss(preds, labels, reduction='none')
    v = (4 / (torch.pi ** 2)) * torch.pow((preds[...,2] - labels[...,2]), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    loss = dloss + alpha * v
    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        iou = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    return loss, iou

def diou_loss(
    preds,
    labels,
    reduction="mean",
    yaw: bool = False,
    eps: float = 1e-7,
):
    preds8 = to_corners(preds)
    labels8 = to_corners(labels)
    iou = calc_iou_array(preds8, labels8).to(DEVICE)
    c2 = enclosing_box_length(preds8, labels8).to(DEVICE) + eps
    d2 = torch.pow((preds[...,0] - labels[...,0]),2) + torch.pow((preds[...,1] - labels[...,1]),2).to(DEVICE)
    
    loss = 1 - iou + (d2 / c2)
    if yaw:
        loss += torch.abs(preds[...,2] - labels[...,2]) / labels[...,2]
    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        iou = iou.mean() if iou.numel() > 0 else 0.0 * iou.sum()
    return loss, iou


def diou_loss_original(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Distance Intersection over Union Loss (Zhaohui Zheng et. al)
    https://arxiv.org/abs/1911.08287
    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # TODO: use torch._assert_async() when pytorch 1.8 support is dropped
    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsct = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsct[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    union = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsct + eps
    iou = intsct / union

    # smallest enclosing box
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)
    diag_len = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + eps

    # centers of boxes
    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2
    distance = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)

    # Eqn. (7)
    loss = 1 - iou + (distance / diag_len)
    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss, iou


def ciou_loss_original(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Complete Intersection over Union Loss (Zhaohui Zheng et. al)
    https://arxiv.org/abs/1911.08287
    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # TODO: use torch._assert_async() when pytorch 1.8 support is dropped
    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsct = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsct[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    union = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsct + eps
    iou = intsct / union

    # smallest enclosing box
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)
    diag_len = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + eps

    # centers of boxes
    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2
    distance = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)

    # width and height of boxes
    w_pred = x2 - x1
    h_pred = y2 - y1
    w_gt = x2g - x1g
    h_gt = y2g - y1g
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    # Eqn. (10)
    loss = 1 - iou + (distance / diag_len) + alpha * v
    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
