import os, cv2
import numpy as np
from glob import glob
from einops import rearrange
import imgaug.augmenters as iaa

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize, Compose, ToPILImage, RandomHorizontalFlip

from data.perlin import rand_perlin_2d_np
from utils import torch_seed

from typing import Union, List, Tuple




class MemSegDataset(Dataset):
    def __init__(self, datadir: str, target: str, train: bool, 
                to_memory: bool=False, resize: Tuple[int, int]=(224, 224),
                texture_source_dir: str=None, structure_grid_size: str=8,
                transparency_range: List[float] =[0.15, 1.],
                perlin_scale: int=6, min_perlin_scale: int=0, perlin_noise_threshold: float=0.5):
        
        # Mode
        self.train     = train
        self.to_memory = to_memory

        # Load image file list
        self.datadir   = datadir
        self.target    = target
        self.file_list = glob(os.path.join(self.datadir, self.target, 'train/*/*' if train else 'test/*/*'))

        # Load texture image file list
        if texture_source_dir:
            self.texture_source_file_list = glob(os.path.join(texture_source_dir, '*/*'))

        # Synthetic anomaly
        if train:
            self.transparency_range     = transparency_range
            self.perlin_scale           = perlin_scale
            self.min_perlin_scale       = min_perlin_scale
            self.perlin_noise_threshold = perlin_noise_threshold
            self.structure_grid_size    = structure_grid_size

        # Transform ndarray into tensor
        self.resize    = resize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Synthetic anomaly switch
        self.anomaly_switch = False

    def __getitem__(self, index):
        file_path = self.file_list[index]

        # Image
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=self.resize)

        # Target
        target = 0 if 'good' in self.file_list[index] else 1

        # Mask
        if 'good' in file_path:
            mask = np.zeros(self.resize, dtype=np.float32)
        else:
            mask = cv2.imread(
                file_path.replace('test', 'ground_truth').replace('.png', '_mask.png'), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, dsize=self.resize).astype(np.bool).astype(np.int)

        # Anomaly Source
        if not self.to_memory and self.train:
            if self.anomaly_switch:
                img, mask = self.generate_anomaly(img=img)
                target = 1
                self.anomaly_switch = False
            else:
                self.anomaly_switch = True

        # Convert ndarray into tensor
        img = self.transform(img)
        mask = torch.Tensor(mask).to(torch.int64)

        return img, mask, target

    def rand_augment(self):
        augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

        aug_idx = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
        aug = iaa.Sequential([
            augmenters[aug_idx[0]],
            augmenters[aug_idx[1]],
            augmenters[aug_idx[2]]
        ])

        return aug

    def generate_anomaly(self, img: np.ndarray) -> List[np.ndarray]:
        """
        STEP 1: Generating Mask
            - Target foreground mask
            - Perlin noise mask
        STEP 2: Generating texture or structure anomaly
            - Texture: load DTD
            - Structure: we first perfrom random adjustment of mirror symmetry, rotation, brightness, saturation,
            and hue on the input image I. Then the preliminary processed image is uniformly divided into a 4x8 grid
            and randomly arranged to obtain the disordered image I.
        STEP 3: Blending image and anomaly source
        """

        ### Step 1: Generating Mask
        # Target foreground mask
        target_foreground_mask = self.generate_target_foreground_mask(img=img)

        # Perlin noise mask
        perlin_noise_mask = self.generate_perlin_noise_mask()

        # Mask
        mask = perlin_noise_mask * target_foreground_mask
        mask_expanded = np.expand_dims(mask, axis=2)

        ### Step 2: Generating texture or structure anomaly
        # Anomaly source
        anomaly_source_img = self.anomaly_source(img=img)

        # Mask anomaly parts
        factor = np.random.uniform(*self.transparency_range, size=1)[0]
        anomaly_source_img = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * img)

        ### Step 3: Blending image and anomaly source
        anomaly_source_img = ((-mask_expanded + 1)*img) + anomaly_source_img

        return (anomaly_source_img.astype(np.uint8), mask)

    def generate_target_foreground_mask(self, img: np.ndarray) -> np.ndarray:
        # Converting RGB into grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Generating binary mask of grayscale image
        _, target_background_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_background_mask = target_background_mask.astype(np.bool).astype(np.int)

        # Inverting mask for foreground mask
        target_foreground_mask = -(target_background_mask - 1)

        return target_foreground_mask

    def generate_perlin_noise_mask(self) -> np.ndarray:
        # Define perlin noise scale
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1, )).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1, )).numpy()[0])

        # Generating perlin noise
        perlin_noise = rand_perlin_2d_np((self.resize[0], self.resize[1]), (perlin_scalex, perlin_scaley))

        # Applying affine transform
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)

        # Making a mask by applying threshold
        mask_noise = np.where(perlin_noise > self.perlin_noise_threshold,
                                np.ones_like(perlin_noise),
                                np.zeros_like(perlin_noise))

        return mask_noise

    
    def anomaly_source(self, img: np.ndarray) -> np.ndarray:
        p = np.random.uniform()
        if p < 0.5:
            # TODO: None texture
            anomaly_source_img = self._texture_source()
        else:
            anomaly_source_img = self._structure_source(img=img)

        return anomaly_source_img

    def _texture_source(self) -> np.ndarray:
        idx = np.random.choice(len(self.texture_source_file_list))
        texture_source_img = cv2.imread(self.texture_source_file_list[idx])
        texture_source_img = cv2.cvtColor(texture_source_img, cv2.COLOR_BGR2RGB)
        texture_source_img = cv2.resize(texture_source_img, dsize=self.resize).astype(np.float32)
        
        return texture_source_img

    def _structure_source(self, img: np.ndarray) -> np.ndarray:
        structure_source_img = self.rand_augment()(image=img)
        
        assert self.resize[0] % self.structure_grid_size == 0, 'structure should be devided by grid size accurately'
        grid_w = self.resize[0] // self.structure_grid_size
        grid_h = self.resize[1] // self.structure_grid_size
        
        structure_source_img = rearrange(
            tensor  = structure_source_img, 
            pattern = '(h gh) (w gw) c -> (h w) gw gh c',
            gw      = grid_w, 
            gh      = grid_h
        )
        disordered_idx = np.arange(structure_source_img.shape[0])
        np.random.shuffle(disordered_idx)

        structure_source_img = rearrange(
            tensor  = structure_source_img[disordered_idx], 
            pattern = '(h w) gw gh c -> (h gh) (w gw) c',
            h       = self.structure_grid_size,
            w       = self.structure_grid_size
        ).astype(np.float32)
        
        return structure_source_img
        
    def __len__(self):
        return len(self.file_list)