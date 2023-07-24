import torch
from med_seg_diff_pytorch import Unet, MedSegDiff
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.dataloader import SatelliteDataset, validate_separator
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import torch


if __name__ == "__main__":

    model = Unet(
        dim=64,
        image_size=224,
        mask_channels=1,  # segmentation has 1 channel
        input_img_channels=3,  # input images have 3 channels
        dim_mults=(1, 2, 4, 8),
    )

    diffusion = MedSegDiff(model, timesteps=1000).cuda()
    transform = A.Compose(
        [
            A.RandomCrop(224, 224),
            A.HorizontalFlip(),
            A.RandomRotate90(p=0.7),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    train_dataset, validate_dataset = validate_separator(
        csv_file="/root/dacon/data/train.csv",
        data_folder="/root/dacon/data",
        transform=transform,
        val_transform=transform,
        validation_ratio=0.9,
    )

    device = "cuda"

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, betas=[0.9, 0.999])

    total_loss = 0
    batch_idx = 0
    for data, target in tqdm(train_dataloader):
        # print('*', end="")

        data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.float32)
        # print("data: ", data)
        optimizer.zero_grad()
        target = target.unsqueeze(1)
        loss = diffusion(target, data)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        # print(loss)
