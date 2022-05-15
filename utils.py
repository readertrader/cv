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

def to_corners(arr):
    points = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]).astype(np.float)
    points *= np.array([arr[3], arr[4]]) / 2
    points = _rotate(points, arr[2])
    points += np.array([arr[0], arr[1]])
    return points

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

def make_batch(batch_size, has_star=True):
    """
    Args:
        batch_size:
            Generates batch_size number of images
        has_star:
            makes batches with only stars if true
            makes randomly generated batches based on synthesize data if false
    """
    imgs, labels = zip(*[synthesize_data(has_star=has_star) for _ in range(batch_size)])
    # Stack into array
    imgs = np.stack(imgs)
    labels = np.stack(labels)
    # Change the yaw from [0,2*pi) to [0,pi) since the bounding box will still be identical
    labels = change_yaw(labels)
    # Convert from numpy to torch
    imgs = torch.from_numpy(np.asarray(imgs, dtype=np.float32))
    labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    imgs = torch.unsqueeze(imgs, 1)
    return imgs, labels


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
    t = Polygon(to_corners(label))
    p = Polygon(to_corners(pred))
    iou = t.intersection(p).area / t.union(p).area
    return iou

def enclosing_box_length(pred, label):
    t = _corners(*label)
    p = _corners(*pred)
    x_t = t[:,0]
    y_t = t[:,1]
    x_p = p[:,0]
    y_p = p[:,1]
    x = np.concatenate((x_t, x_p))
    y = np.concatenate((y_p, y_t))
    top = min(y)
    bottom = max(y)
    left = min(x)
    right = max(x)
    width = abs(right - left)
    height = abs(bottom - top)
    return height**2 + width**2

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

def unnormalize_tensor(label, img_size=200):
    center = label[:, 0:2] * img_size
    yaw = label[:, 2] * np.pi
    yaw = yaw.reshape((yaw.shape[0], 1))
    size = label[:,3:] * img_size
    original = np.concatenate((center, yaw, size), axis=1)
    return original
