from selectors import EpollSelector
from statistics import median_high
from typing import NamedTuple, Optional

import numpy as np
import torch
from shapely.geometry import Polygon
from skimage.draw import line, polygon_perimeter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Star(NamedTuple):
    points: np.ndarray
    label: np.array


class StarImage(NamedTuple):
    image: np.ndarray
    label: np.ndarray


def _rotate(points: np.ndarray, theta: float) -> np.ndarray:
    return points @ np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def _corners(pos_x: float, pos_y: float, yaw: float, width: float, height: float) -> np.ndarray:
    points = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]).astype(np.float)
    points *= np.array([width, height]) / 2
    points = _rotate(points, yaw)
    points += np.array([pos_x, pos_y])
    return points


def _star(
    pos: np.asarray,
    yaw: float,
    side_len: float,
    i_stretch: float,
    j_stretch: float,
    roundness: float,
) -> Star:
    """
    Returns:
        Star:
            the polygon points of the star and corresponding label
    """

    # normal star (centered at origin, side length 1) with given roundness
    points = np.array(
        [
            [1, 0],
            [roundness, roundness],
            [0, 1],
            [-roundness, roundness],
            [-1, 0],
            [-roundness, -roundness],
            [0, -1],
            [roundness, -roundness],
        ]
    )

    # stretch, rotate, translate
    width_height = (side_len) * np.array([i_stretch, j_stretch])
    points *= width_height / 2
    points = _rotate(points, yaw)
    points += pos

    return Star(points, np.asarray([*pos, yaw, *(width_height)]))


def _clipped_normal(mean, var, clip_low, clip_high, size=1) -> np.ndarray:
    return np.clip(np.random.normal(mean, var, size=size), clip_low, clip_high)


def _pos(s: float) -> np.ndarray:
    return np.random.randint(40, s - 40, size=2)


def _yaw() -> float:
    return np.random.uniform(0, 2 * np.pi)


def _side_len() -> int:
    return np.random.randint(70, 90)


def _width_stretch() -> float:
    return np.clip(np.random.normal(0.7, 0.4 / 3), 0.6, 0.8)


def _height_stretch() -> float:
    return np.clip(np.random.normal(1.3, 0.4 / 3), 1.10, 1.3)


def _roundness() -> float:
    return np.clip(np.random.normal(0.25, 0.2 / 3), 0.10, 0.40)

def synthesize_data(
    has_star: bool = None,
    noise_level: float = 0.2,
    n_lines: int = 8,
    image_size: int = 200,
) -> StarImage:
    """
    Args:
        has_star (bool, optional):
            Whether a star is included. Defaults to None, in which case it is
            True with frequency 0.8.
        noise_level (float, optional):
            Amount of noise. Defaults to 0.2.
        n_lines (int, optional): Number of extra lines. Defaults to 12.
        image_size (int, optional): Size of synthesized image. Defaults to 200.

    Returns:
        StarImage:
            Synthesized image and corresponding label. As described in the
            README, the label consists of the bounding box's x coordinate,
            y coordinate, yaw, width, and height; when there is no star,
            it is an array of np.nan's.
    """
    if has_star is None:
        has_star = np.random.choice([True, False], p=(0.8, 0.2))

    image = _clipped_normal(0.0, noise_level, 0, 1, size=(image_size, image_size))

    def noisy_draw(x, y):
        valid = (x >= 0) & (x < image_size) & (y >= 0) & (y < image_size)
        image[x[valid], y[valid]] = _clipped_normal(0.25, noise_level, 0, 1, size=np.sum(valid))

    # star (or no-star)
    if has_star:
        points, label = _star(
            _pos(image_size), _yaw(), _side_len(), _width_stretch(), _height_stretch(), _roundness()
        )
        noisy_draw(*polygon_perimeter(points[:, 0], points[:, 1]))
    else:
        label = np.full(5, np.nan)

    # lines
    for _ in range(n_lines):
        noisy_draw(
            *line(*np.random.randint(-image_size // 2, image_size + image_size // 2, size=4))
        )

    return StarImage(image.T, label)  # coordinate transform


def score_iou(pred: np.ndarray, label: np.ndarray) -> Optional[float]:
    assert (
        pred.size == 5 and label.size == 5
    ), "Preds & labels should have length 5. Use nan's for no-star."

    pred_no_star = np.any(np.isnan(pred))
    label_no_star = np.any(np.isnan(label))

    if not label_no_star and not pred_no_star:
        # true positive
        t = Polygon(_corners(*label))
        p = Polygon(_corners(*pred))
        iou = t.intersection(p).area / t.union(p).area
        return iou
    elif (label_no_star and not pred_no_star) or (not label_no_star and pred_no_star):
        # false positive or false negative
        return 0
    elif label_no_star and pred_no_star:
        # true negative
        return None
    else:
        raise NotImplementedError

def calc_iou(pred, label):
    t = Polygon(_corners(*label))
    p = Polygon(_corners(*pred))
    iou = t.intersection(p).area / t.union(p).area
    return iou

def calc_iou_array(pred, label):
    B = pred.size()[0]
    arr = torch.zeros(B).to(pred.device)
    for i in range(B):
        p = Polygon(pred[i][0])
        t = Polygon(label[i][0])
        arr[i] = t.intersection(p).area / t.union(p).area
    return arr.view([-1,1])

def to_corners(arr):
    batch = arr.size()[0]
    x = torch.FloatTensor([1, -1, -1, 1]).unsqueeze(0).unsqueeze(0).to(arr.device)
    y = torch.FloatTensor([1, 1, -1, -1]).unsqueeze(0).unsqueeze(0).to(arr.device)
    x1 = arr[..., 0:1]
    y1 = arr[..., 1:2]
    yaw = arr[..., 2:3]
    w = arr[..., 3:4]
    h = arr[...,4:5]
    x = x*w/2
    y = y*h/2
    points = torch.stack([x, y], dim=-1)
    sin = torch.sin(yaw)
    cos = torch.cos(yaw)
    row1 = torch.cat([cos, -sin], dim=-1)
    row2 = torch.cat([sin, cos], dim=-1)  
    rot_T = torch.stack([row1, row2], dim=-2) 
    rotated = torch.bmm(points.view([-1, 4, 2]), rot_T.view([-1, 2, 2]))
    rotated = rotated.view([batch, -1, 4, 2])  
    rotated[..., 0] += x1
    rotated[..., 1] += y1
    return rotated

def create_fpn_label(label, img_size=200):
    no_star = np.any(np.isnan(label))
    if no_star:
        return np.concatenate([np.array([0]), normalize(label, img_size=img_size)])
    else:
        return np.concatenate([np.array([1]), normalize(label,img_size=img_size)])

def normalize(label, img_size=200):
    normalized_center = label[0:2] / img_size
    if label[2] >= np.pi:
        label[2] = label[2] - np.pi
    normalized_yaw = [label[2] / np.pi]
    normalized_size = label[3:] / img_size
    normalized = np.concatenate((normalized_center, normalized_yaw, normalized_size))
    return normalized

def unnormalize(label, img_size=200):
    center = label[0:2] * img_size
    yaw = [label[2] * np.pi]
    size = label[3:] * img_size
    original = np.concatenate((center, yaw, size))
    return original

# Need a method to unnormalize tensors to maintain Variable integrity to compute loss.backwards
def unnormalize_tensor(label, img_size=200):
    x = label[:, 0] * img_size
    y = label[:,1] * img_size
    yaw = label[:, 2] * np.pi
    w = label[:,3] * img_size
    h = label[:,4] * img_size
    original = torch.stack([x,y, yaw, w,h], dim=1)
    return original

def norm_classifier(label):
    no_star = np.any(np.isnan(label))
    if no_star:
        return np.array([0])
    else:
        return np.array([1])
