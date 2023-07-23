from models.unet.UNet3Plus import UNet3Plus_DeepSup
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

with open("models/unet3plus_deepsup.yaml") as f:
    cfg = yaml.safe_load(f)

device = "cuda"

transform = A.Compose([A.Normalize(), ToTensorV2()])

model = UNet3Plus_DeepSup(**cfg["model"])
filename = "/root/dacon/models/ckpt/unet3plus_deepsup/Unet3plus_deepsup_shallower_249_07-23-16:21"
model.load_state_dict(torch.load(filename))
model.to(device).eval()

test_dataset = SatelliteDataset(data="data/test.csv", transform=transform, infer=True)
test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

with torch.no_grad():
    model.eval()
    result = []
    for images in tqdm(test_dataloader):
        images = images.float().to(device)

        outputs = model(images)
        output = torch.concat(outputs, dim=1).mean(dim=1).cpu().numpy()
        # print(output.shape)
        masks = (output > 0.5).astype(np.uint8)  # Threshold = 0.35

        for i in range(len(images)):
            mask_rle = rle_encode(masks[i])
            if mask_rle == "":  # 예측된 건물 픽셀이 아예 없는 경우 -1
                result.append(-1)
            else:
                result.append(mask_rle)

submit = pd.read_csv("data/sample_submission.csv")
submit["mask_rle"] = result

filename = cfg["model-name"] + "_submit.csv"
submit.to_csv(f"data/{filename}", index=False)
