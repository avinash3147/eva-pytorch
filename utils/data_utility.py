"""
contains all methods related to data
"""
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import datasets, transforms


def download_train_data(train_transforms):
    """
    Downloads Train Data
    Args:
        train_transforms: Applies transformations on train data

    Returns: train data

    """
    train_data = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=train_transforms
    )
    return train_data


def download_test_data(test_transforms):
    """
    Download Test  Data
    Args:
        test_transforms: Transformations to be applied on test data

    Returns: test data

    """
    test_data = datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=test_transforms
    )
    return test_data


def load_train_data(train_data, **data_loader_args):
    """
    Load Train Data
    Args:
        train_data: train data
        **data_loader_args: additional params used while loading dataa

    Returns: train loader

    """
    train_loader = torch.utils.data.DataLoader(
        train_data,
        **data_loader_args
    )
    return train_loader


def load_test_data(test_data, **data_loader_args):
    """
    Load Test Data
    Args:
        test_data: test data
        **data_loader_args: additional params used while using loading data

    Returns: test loader

    """
    test_loader = torch.utils.data.DataLoader(
        test_data,
        **data_loader_args
    )
    return test_loader


def train_data_transformation():
    """
    Set of transformations to be applied on train data
    Returns: list of transformations

    """
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.3),
        A.RandomCrop(32, padding=4),
        A.Cutout(num_holes=1, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, interpolation=cv2.INTER_LINEAR,
                           border_mode=cv2.BORDER_REFLECT_101, always_apply=False, p=0.5),
        A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16,
                        min_width=16, fill_value=(0.4914, 0.4822, 0.4465), mask_fill_value = None),
        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ToTensorV2(),
    ])
    return train_transforms


def test_data_transformation():
    """
    Set of transforms to be applied on test data
    Returns: list of transforms

    """
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return test_transforms



def get_data_loader_args(cuda):
    return dict(shuffle=True, batch_size=128, num_workers=1, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

