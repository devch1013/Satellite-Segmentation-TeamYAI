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

with open("models/MAResUnet.yaml") as f:
    cfg = yaml.safe_load(f)

device = "cuda"

transform = A.Compose([A.Normalize(), ToTensorV2()])

model = MAResUNet(**cfg["model"])
filename = "/root/dacon/models/ckpt/MAResUnet/MAResUnet_msssim_retrain_15_07-24-17:45"
model.load_state_dict(torch.load(filename))
model.to(device).eval()

test_dataset = SatelliteDataset(data="data/test.csv", transform=transform, infer=True)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

with torch.no_grad():
    model.eval()
    result = []
    for images in tqdm(test_dataloader):
        images = images.float().to(device)

        outputs = model(images)
        masks = outputs.cpu().numpy()
        masks = np.squeeze(masks, axis=1)
        masks = (masks > 0.5).astype(np.uint8)  # Threshold = 0.35

        for i in range(len(images)):
            mask_rle = rle_encode(masks[i])
            if mask_rle == "":  # 예측된 건물 픽셀이 아예 없는 경우 -1
                result.append(-1)
            else:
                result.append(mask_rle)

submit = pd.read_csv("data/sample_submission.csv")
submit["mask_rle"] = result

filename = "MAResUnet" + "_submit.csv"
submit.to_csv(f"data/{filename}", index=False)
