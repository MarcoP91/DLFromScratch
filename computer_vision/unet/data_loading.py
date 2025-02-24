import logging
import numpy as np
import torch
import cv2  # Import OpenCV
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image


def load_image(filename, is_mask=False):
    """Loads an image with OpenCV, handling both regular images and masks correctly."""
    ext = Path(filename).suffix.lower()

    if ext == '.npy':
        return np.load(filename)
    elif ext in ['.pt', '.pth']:
        return torch.load(filename).numpy()
    elif ext == '.gif' and is_mask:
        # Use Pillow to load the first frame of GIF masks
        with Image.open(filename) as img:
            img = img.convert("L")  # Convert to grayscale
            return np.array(img)
    else:
        # Use IMREAD_GRAYSCALE for masks, IMREAD_COLOR for images
        flag = cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR
        img = cv2.imread(str(filename), flag)

        if img is None:
            raise FileNotFoundError(f"🚨 OpenCV failed to load: {filename}")

        # Convert BGR to RGB for images (but not for grayscale masks)
        if not is_mask:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img


def unique_mask_values(idx, mask_dir, mask_suffix):
    """Find unique values in a segmentation mask."""
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = load_image(mask_file, is_mask=True)
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def pad_to_multiple_of_16(image):
        """Pad images (RGB) and masks (grayscale) to ensure dimensions are multiples of 16."""
        if len(image.shape) == 2:  # Grayscale mask (H, W)
            h, w = image.shape
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16

            # Pad grayscale mask with 0 (black)
            padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        else:  # RGB image (H, W, C)
            h, w, c = image.shape
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16

            # Pad RGB image with (0, 0, 0) (black)
            padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        return padded_image


    @staticmethod
    def preprocess(mask_values, img, scale, is_mask):
        """Resize, pad, and normalize images/masks."""

        # Pad image to be a multiple of 16
        img = BasicDataset.pad_to_multiple_of_16(img)
        h, w = img.shape[:2]
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'

        # Resize image using OpenCV
        img = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            return mask

        else:
            # Convert HWC (OpenCV format) to CHW (PyTorch format)
            img = img.transpose((2, 0, 1))

            # Normalize if necessary
            if img.max() > 1:
                img = img / 255.0  # Convert to range [0,1]

            return img

    def __getitem__(self, idx):
        """Load an image and its corresponding mask, preprocess them, and return as tensors."""
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

        mask = load_image(mask_file[0], is_mask=True)
        img = load_image(img_file[0], is_mask=False)

        assert img.shape[:2] == mask.shape[:2], \
            f'Image and mask {name} should be the same size, but are {img.shape[:2]} and {mask.shape[:2]}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
