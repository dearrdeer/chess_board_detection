import numpy as np
from dataset.dataset import ChessBoardDataset, ChessTestBoardDataset
import albumentations as A
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import cv2


def prepare_datasets(path_to_folder: str, split_size:int = 0.8):
    xtrain = np.load(f'{path_to_folder}/xtrain.npy')
    ytrain = np.load(f'{path_to_folder}/ytrain.npy')
    xtest = np.load(f'{path_to_folder}/xtest.npy')

    transforms = A.Compose([A.Normalize(mean=0, std=1)])
    full_ds = ChessBoardDataset(xtrain, ytrain, transforms)
    dataset_size = xtrain.shape[0]
    train_ds, val_ds = random_split(full_ds, [int(dataset_size*split_size), dataset_size-int(dataset_size*split_size)])

    test_ds = ChessTestBoardDataset(xtest, transforms)

    return train_ds, val_ds, test_ds


def show_marked_image(img: np.array, pts: np.array, polygon_color:tuple = (255, 0, 0)):
    shape = img.shape
    cv_points = []
    i = 0

    while i < len(pts):
        point = [int(shape[0] * pts[i]), int(shape[1] * pts[i + 1])]
        cv_points.append(point)
        i += 2

    gray_img = np.reshape(img, (256, 256))
    rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    marked_img = cv2.polylines(rgb_img, [np.array(cv_points)], True, polygon_color)
    plt.imshow(marked_img.astype(np.uint32))
