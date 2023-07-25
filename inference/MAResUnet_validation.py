from models.MAResUnet.MAResUnet import MAResUNet
import yaml
import torch
from tqdm import tqdm
from utils.dataloader import SatelliteDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import numpy as np
from utils.utils import rle_encode
import pandas as pd
from utils.dataloader import validate_separator
from utils.losses.diceLoss import dice_coeff_batch


with open("models/MAResUnet.yaml") as f:
    cfg = yaml.safe_load(f)
device = "cuda"

transform = A.Compose(
    [
        A.augmentations.crops.transforms.CenterCrop(224, 224, p=1),
        A.Normalize(),
        ToTensorV2(),
    ]
)

model = MAResUNet(**cfg["model"])
filename = "/root/dacon/models/ckpt/MAResUnet/MAResUnet_msssim_541_07-24-16:40"
model.load_state_dict(torch.load(filename))
model.to(device).eval()

train_dataset, validate_dataset = validate_separator(
    csv_file="/root/dacon/data/train.csv",
    data_folder="/root/dacon/data",
    transform=transform,
    val_transform=transform,
    validation_ratio=0.9,
)

test_dataloader = DataLoader(validate_dataset, batch_size=16, shuffle=True)

with torch.no_grad():
    model.eval()
    result = []
    total_dice_score = 0
    for images, target in tqdm(test_dataloader):
        images, target = images.to(device, dtype=torch.float32), target.to(
            device, dtype=torch.float32
        )

        outputs = model(images)[0]
        masks = np.squeeze(outputs, axis=1)
        # masks = torch.sigmoid(outputs).cpu().numpy()
        # masks = np.squeeze(masks, axis=1)
        masks = (masks > 0.5).float()  # Threshold = 0.35

        total_dice_score += dice_coeff_batch(input=masks, target=target.unsqueeze(dim=1)).item()


print("dice score: ", total_dice_score / len(test_dataloader))

# submit = pd.read_csv('data/train.csv')
# submit['mask_rle'] = result

# filename = "train_inference.csv"
# submit.to_csv(f'data/{filename}', index=False)

# with torch.no_grad():
#     model.eval()
#     result = []
#     for images in tqdm(test_dataloader):
#         images = images.float().to(device)

#         outputs = model(images)
#         masks = torch.sigmoid(outputs).cpu().numpy()
#         masks = np.squeeze(masks, axis=1)
#         masks = (masks > 0.5).astype(np.uint8) # Threshold = 0.35

#         for i in range(len(images)):
#             mask_rle = rle_encode(masks[i])
#             if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
#                 result.append(-1)
#             else:
#                 result.append(mask_rle)

# submit = pd.read_csv('data/train.csv')
# submit['mask_rle'] = result

# filename = "train_inference.csv"
# submit.to_csv(f'data/{filename}', index=False)
