from train.train_class import Trainer
from models.backboned_unet.unet import Unet
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.dataloader import SatelliteDataset, validate_separator, ValidateDataset
import torch

model_name = "unet3plus_deepsup"
root_dir = "/root/dacon"
device = "cuda"

if __name__ == "__main__":
    model = Unet(backbone_name="resnet152", n_classes=1)
    filename = "/root/dacon/models/ckpt/checkpoint_resnet152_epoch1219.pth"
    # filename = "/root/dacon/models/ckpt/checkpoint_resnet152_epoch282.pth"
    state_dict = torch.load(filename, map_location=device)
    mask_values = state_dict.pop("mask_values", [0, 1])
    model.load_state_dict(state_dict)

    transform = A.Compose(
        [
            A.augmentations.crops.transforms.CenterCrop(
                height=224, width=224, always_apply=True, p=1
            ),
            # A.augmentations.transforms.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
            # A.Normalize(),
            ToTensorV2(),
        ]
    )

    test_dataset = SatelliteDataset(data="data/test.csv", transform=transform, infer=True)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    validate_dataset = ValidateDataset(transform=validate_transform)

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
