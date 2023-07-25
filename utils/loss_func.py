import numpy as np
import pandas as pd
from typing import List, Union
from joblib import Parallel, delayed
import torch
from utils.losses.bceLoss import BCE_loss
from utils.losses.diceLoss import dice_coeff_batch
from utils.losses.msssimLoss import msssim
from utils.losses.focal_loss import focal_loss
from utils.losses.hausdorff_loss import hausdorff_loss
import torch.nn.functional as F


def rle_decode(mask_rle: Union[str, int], shape=(224, 224)) -> np.array:
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """

    if mask_rle == -1:
        return np.zeros(shape)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def dice_coeff(
    input: torch.Tensor,
    target: torch.Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(
    input: torch.Tensor,
    target: torch.Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: torch.Tensor, target: torch.Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def bce_dice(input: torch.Tensor, target: torch.Tensor):
    # print("loss input: ", input)
    dice = 1 - dice_coeff_batch(input, target)
    bce = BCE_loss(input, target)
    loss_dict = {
        "Dice Loss": dice,
        "Binary Cross Entropy": bce,
    }
    # return total_loss  # , dice, bce, msssim_loss
    return loss_dict


def hybrid_seg_loss(input: torch.Tensor, target: torch.Tensor):
    # print("loss input: ", input)
    dice = 1 - dice_coeff_batch(input, target)
    bce = BCE_loss(input, target)
    msssim_loss = 1 - msssim(input, target, normalize=True)
    total_loss = dice + bce + msssim_loss
    # print(dice, bce, msssim_loss)
    loss_dict = {
        "Dice Loss": dice,
        "Binary Cross Entropy": bce,
        "MSSIM Loss": msssim_loss,
    }
    # return total_loss  # , dice, bce, msssim_loss
    return loss_dict


def hybrid_seg_loss_focal(input: torch.Tensor, target: torch.Tensor):
    # print("loss input: ", input)
    dice = 1 - dice_coeff_batch(input, target)
    bce = focal_loss(input, target)
    msssim_loss = 1 - msssim(input, target, normalize=True)
    total_loss = dice + bce + msssim_loss
    # print(dice, bce, msssim_loss)
    loss_dict = {
        "Dice Loss": dice,
        "Focal Loss": bce,
        "MSSIM Loss": msssim_loss,
    }
    # return total_loss  # , dice, bce, msssim_loss
    return loss_dict


def dice_bce_hausdorff(input: torch.Tensor, target: torch.Tensor):
    # print("loss input: ", input)
    dice = 1 - dice_coeff_batch(input, target)
    bce = focal_loss(input, target)
    # print(target)
    hd_loss = hausdorff_loss(input, target.to(dtype=torch.long))
    total_loss = dice + bce + hd_loss
    # print(dice, bce, hausdorff_loss)
    loss_dict = {
        "Dice Loss": dice,
        "Focal Loss": bce,
        "HD Loss": hd_loss,
    }
    # return total_loss  # , dice, bce, hausdorff_loss
    return loss_dict
