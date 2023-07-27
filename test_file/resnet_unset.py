import torch
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import numpy as np
from utils.utils import rle_encode
import pandas as pd
import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from backboned_Unet.backboned_unet import Unet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib


def get_args():
    parser = argparse.ArgumentParser(description="Predict masks from input images")
    parser.add_argument(
        "--model",
        "-m",
        default="/home/work/Dacon-YAI/claire/resnet152_ckpt/checkpoint_resnet152_epoch742.pth",
        metavar="FILE",
        help="Specify the file in which the model is stored",
    )
    parser.add_argument(
        "--input", "-i", metavar="INPUT", nargs="+", help="Filenames of input images"
    )
    parser.add_argument(
        "--output", "-o", metavar="OUTPUT", nargs="+", help="Filenames of output images"
    )
    parser.add_argument(
        "--viz", "-v", action="store_true", help="Visualize the images as they are processed"
    )
    parser.add_argument("--no-save", "-n", action="store_true", help="Do not save the output masks")
    parser.add_argument(
        "--mask-threshold",
        "-t",
        type=float,
        default=0.5,
        help="Minimum probability value to consider a mask pixel white",
    )
    parser.add_argument(
        "--scale", "-s", type=float, default=1, help="Scale factor for the input images"
    )
    parser.add_argument(
        "--bilinear", action="store_true", default=False, help="Use bilinear upsampling"
    )
    parser.add_argument("--classes", "-c", type=int, default=1, help="Number of classes")

    return parser.parse_args()


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(
        BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False)
    ).permute(2, 0, 1)
    img = img.unsqueeze(0)
    # print(img.size())
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode="bilinear")
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy().astype(np.uint8)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = get_args()
# file_path = args.input
file_path = "/home/work/Dacon-YAI/data/test_img"
print(file_path)
in_files = os.listdir(file_path)

net = Unet(backbone_name="resnet152", n_classes=1)
# model = Unet(backbone_name='resnet152', n_classes=1)
# print(net)
state_dict = torch.load(args.model, map_location=device)
mask_values = state_dict.pop("mask_values", [0, 1])
net.load_state_dict(state_dict)
net = net.to(device)
num_dataset = len(in_files)
data = pd.read_csv("/home/work/Dacon-YAI/data/test.csv")
# img_path = data.iloc[idx, 1]
logging.info("Model loaded!")

result = []
for idx in tqdm(range(num_dataset)):
    # print(f'Predicting image {filename} ...')
    img_path = data.iloc[idx, 1]
    img = Image.open("/home/work/Dacon-YAI/data" + img_path[1:])
    mask = predict_img(
        net=net,
        full_img=img,
        scale_factor=args.scale,
        out_threshold=args.mask_threshold,
        device=device,
    )
    # print(mask.shape)
    # matplotlib.image.imsave('name.jpg', mask)
    # for j in range(len(mask)):
    pix_count = np.count_nonzero(mask == 1)
    mask_rle = rle_encode(mask)
    if mask_rle == "" or pix_count < 10:  # 예측된 건물 픽셀이 아예 없는 경우 -1
        result.append(-1)
    else:
        result.append(mask_rle)

submit = pd.read_csv("/home/work/Dacon-YAI/data/sample_submission.csv")
submit["mask_rle"] = result

filename = "submit_resnet152_unet_742.csv"
submit.to_csv(f"/home/work/Dacon-YAI/data/{filename}", index=False)
