import torch
from utils import DEVICE
from torch import nn
import torch.nn.functional as F

# https://arxiv.org/abs/1911.08299
def modulated_loss(pred, label):
    x1, x2 = pred[:, 0], label[:, 0]
    y1, y2 = pred[:, 1], label[:, 1]
    yaw1, yaw2 = pred[:, 2], label[:, 2]
    w1, w2 = pred[:, 3], label[:, 3]
    h1, h2 = pred[:, 4], label[:, 4]
    lcp = torch.abs(x1 - x2) + torch.abs(y1 - y2)
    lmr = torch.min(
        lcp + torch.abs(w1 - w2) + torch.abs(h1 - h2) + torch.abs(yaw1 - yaw2),
        lcp + torch.abs(w1 - h2) + torch.abs(h1 - w2) + torch.abs(torch.pi/2 - torch.abs(yaw1 - yaw2))
    )
    return lmr

# KFIOU implementation from https://github.com/open-mmlab/mmrotate/blob/main/mmrotate/models/losses/kf_iou_loss.py
def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.
    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).
    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 3:5].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 2]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma

def kfiou_loss(pred,
               target,
               pred_decode=None,
               targets_decode=None,
               fun=None,
               beta=1.0 / 9.0,
               eps=1e-6):
    """Kalman filter IoU loss.
    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        pred_decode (torch.Tensor): Predicted decode bboxes.
        targets_decode (torch.Tensor): Corresponding gt decode bboxes.
        fun (str): The function applied to distance. Defaults to None.
        beta (float): Defaults to 1.0/9.0.
        eps (float): Defaults to 1e-6.
    Returns:
        loss (torch.Tensor)
    """
    xy_p = pred[:, :2]
    xy_t = target[:, :2]
    _, Sigma_p = xy_wh_r_2_xy_sigma(pred_decode)
    _, Sigma_t = xy_wh_r_2_xy_sigma(targets_decode)

    # Smooth-L1 norm
    diff = torch.abs(xy_p - xy_t)
    xy_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                          diff - 0.5 * beta).sum(dim=-1)
    Vb_p = 4 * Sigma_p.det().sqrt()
    Vb_t = 4 * Sigma_t.det().sqrt()
    K = Sigma_p.bmm((Sigma_p + Sigma_t).inverse())
    Sigma = Sigma_p - K.bmm(Sigma_p)
    Vb = 4 * Sigma.det().sqrt()
    Vb = torch.where(torch.isnan(Vb), torch.full_like(Vb, 0), Vb)
    KFIoU = Vb / (Vb_p + Vb_t - Vb + eps)

    if fun == 'ln':
        kf_loss = -torch.log(KFIoU + eps)
    elif fun == 'exp':
        kf_loss = torch.exp(1 - KFIoU) - 1
    else:
        kf_loss = 1 - KFIoU

    loss = (xy_loss + kf_loss).clamp(0)

    return loss


class KFLoss(nn.Module):
    """Kalman filter based loss.
    Args:
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        reduction (str, optional): The reduction method of the
            loss. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.
    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 fun='none',
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super(KFLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert fun in ['none', 'ln', 'exp']
        self.fun = fun
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                pred_decode=None,
                targets_decode=None,
                reduction_override=None,
                **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): Predicted convexes.
            target (torch.Tensor): Corresponding gt convexes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            pred_decode (torch.Tensor): Predicted decode bboxes.
            targets_decode (torch.Tensor): Corresponding gt decode bboxes.
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        Returns:
            loss (torch.Tensor)
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)

        return kfiou_loss(
            pred,
            target,
            fun=self.fun,
            weight=weight,
            avg_factor=avg_factor,
            pred_decode=pred_decode,
            targets_decode=targets_decode,
            reduction=reduction,
            **kwargs) * self.loss_weight

def kfiou_loss_fpn(pred,
               target,
               pred_decode=None,
               targets_decode=None,
               fun=None,
               beta=1.0 / 9.0,
               eps=1e-6):

    no_star = torch.nonzero(target[:, 0] == 0, as_tuple=True)
    loss_clf = F.binary_cross_entropy_with_logits(pred[:,0], target[:,0], reduction='none')
    print(loss_clf.mean())
    loss_regressor = kfiou_loss(pred, target, pred_decode, targets_decode)
    loss_regressor[no_star] = 0
    loss = loss_clf + loss_regressor

    return loss, loss_clf, loss_regressor