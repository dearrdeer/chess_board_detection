import cv2
from torch.utils.data import Dataset
import numpy as np
import torch


class ChessBoardDataset(Dataset):
    def __init__(self, images: np.array, board_points: np.array, transforms=None):
        self.images = images
        self.points = board_points
        self.transforms = transforms

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # make images to have 3 channels
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        pts = self.points[idx]
        pts = torch.from_numpy(pts)
        return img, pts


class ChessTestBoardDataset(Dataset):
    def __init__(self, images: np.array, transforms=None):
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        return img
