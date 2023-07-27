from train.train_class_mask2mask import Trainer
from models.unet.mask2mask import Mask2Mask
from models.MAResUnet.MAResUnet import MAResUNet
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.dataloader2 import SatelliteDataset, validate_separator, ValidateDataset
import torch
import yaml

model_name = "MAResUnet_mask2mask"
root_dir = "/root/dacon"
device = "cuda"

if __name__ == "__main__":
    with open("models/MAResUnet.yaml") as f:
        cfg = yaml.safe_load(f)
    train_class = Trainer(base_dir=root_dir, config_dir=f"models/{model_name}.yaml")
    # train_class.set_model(
    #     Unet(backbone_name="resnet152", n_classes=1),
    #     state_dict="/root/dacon/models/ckpt/checkpoint_epoch740.pth",
    # )
    main_model = MAResUNet(**cfg["model"])
    model = Mask2Mask(main_model=main_model)
    filename = "/root/dacon/models/ckpt/MAResUnet_mask2mask/MAResUnet_mask2mask_lossweight_109_07-27-19:14_0.8364231565168926"
    state_dict = torch.load(filename, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    train_class.set_pretrained_model(model)

    transform = A.Compose(
        [
            A.RandomCrop(224, 224),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
            A.RandomGamma(gamma_limit=(90, 110)),
            A.RandomRotate90(p=0.7),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    validate_transform = A.Compose(
        [
            A.augmentations.crops.transforms.CenterCrop(
                height=224, width=224, always_apply=True, p=1
            ),
            # A.augmentations.transforms.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
            A.Normalize(max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )

    train_dataset, validate_dataset = validate_separator(
        csv_file="data/train.csv",
        data_folder="data",
        transform=transform,
        val_transform=validate_transform,
        validation_ratio=0.95,
    )
    validate_dataset = ValidateDataset(transform=validate_transform)
    train_class.set_train_dataloader(dataset=train_dataset)
    train_class.set_validation_dataloader(dataset=validate_dataset)
    # train_class.enable_ckpt(f"models/ckpt/{model_name}")
    # train_class.enable_tensorboard(save_image_log=True)
    train_class.validate()
