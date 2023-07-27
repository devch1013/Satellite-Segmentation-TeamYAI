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
    train_class = Trainer(base_dir=root_dir, config_dir=f"models/{model_name}.yaml")
    # train_class.set_model(
    #     Unet(backbone_name="resnet152", n_classes=1),
    #     state_dict="/root/dacon/models/ckpt/checkpoint_epoch740.pth",
    # )

    model = Unet(backbone_name="resnet152", n_classes=1)
    filename = "/root/dacon/models/ckpt/checkpoint_resnet152_epoch1257.pth"
    # filename = "/root/dacon/models/ckpt/checkpoint_resnet152_epoch282.pth"
    state_dict = torch.load(filename, map_location=device)
    mask_values = state_dict.pop("mask_values", [0, 1])
    model.load_state_dict(state_dict)

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
            # A.Normalize(),
            ToTensorV2(),
        ]
    )

    train_dataset, validate_dataset = validate_separator(
        csv_file="data/train.csv",
        data_folder="data",
        transform=transform,
        val_transform=validate_transform,
        validation_ratio=0.9,
    )
    validate_dataset = ValidateDataset(transform=validate_transform)
    train_class.set_train_dataloader(dataset=train_dataset)
    train_class.set_validation_dataloader(dataset=validate_dataset)
    # train_class.enable_ckpt(f"models/ckpt/{model_name}")
    # train_class.enable_tensorboard(save_image_log=True)
    train_class.validate()
