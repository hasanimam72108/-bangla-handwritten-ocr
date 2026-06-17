import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def clahe_equalization(img: np.ndarray, clip_limit: float = 2.0, grid_size: tuple = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)


def denoise(img: np.ndarray, h: float = 10.0) -> np.ndarray:
    return cv2.fastNlMeansDenoising(img, None, h, 7, 21)


def resize_to_fixed_height(img: np.ndarray, target_height: int = 384) -> np.ndarray:
    h, w = img.shape[:2]
    aspect = w / h
    target_width = int(target_height * aspect)
    target_width = max(target_width, target_height)
    return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)


def pad_to_square(img: np.ndarray, target_size: int = 384, pad_val: int = 255) -> np.ndarray:
    h, w = img.shape[:2]
    if len(img.shape) == 2:
        square = np.full((target_size, target_size), pad_val, dtype=np.uint8)
    else:
        square = np.full((target_size, target_size, img.shape[2]), pad_val, dtype=np.uint8)

    if h > target_size or w > target_size:
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]

    y_off = (target_size - h) // 2
    x_off = (target_size - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = img
    return square


def preprocess_pipeline(img: np.ndarray, target_size: int = 384) -> np.ndarray:
    img = to_grayscale(img)
    img = clahe_equalization(img)
    img = denoise(img)
    img = resize_to_fixed_height(img, target_size)
    img = pad_to_square(img, target_size)
    return img


class ImageTransform:
    def __init__(self, target_size: int = 384, augment: bool = False):
        self.target_size = target_size
        self.augment = augment
        base_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        if augment:
            base_transforms = [
                transforms.RandomRotation(degrees=5, fill=255),
                transforms.RandomAffine(
                    degrees=3,
                    translate=(0.03, 0.03),
                    scale=(0.92, 1.08),
                    shear=8,
                    fill=255,
                ),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.4, fill=255),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=255),
            ] + base_transforms
        self.transform = transforms.Compose(base_transforms)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((self.target_size, self.target_size), Image.BICUBIC)
        return self.transform(img)
