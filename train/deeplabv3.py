from train.train_class import Trainer
from models.deeplabv3.modeling import deeplabv3plus_resnet101
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.dataloader import SatelliteDataset, validate_separator, ValidateDataset
import torch

model_name = "deeplabv3plus_resnet101"
root_dir = "/root/dacon"


if __name__ == "__main__":
    train_class = Trainer(base_dir=root_dir, config_dir=f"models/{model_name}.yaml")
    # train_class.set_model(
    #     UNet3Plus_DeepSup,
    #     # state_dict="/home/work/Dacon-YAI/devch/models/ckpt/unet3plus_deepsup_cgm/Unet3plus_deepsup_cgm_shallow_266_07-21-14:22",
    # )

    model = deeplabv3plus_resnet101(1, pretrained_backbone=True)
    filename = "/root/dacon/models/ckpt/deeplabv3plus_resnet101/deeplabv3plus_resnet101_287_07-27-13:31_0.662719850859991"
    model.load_state_dict(torch.load(filename))

    train_class.set_pretrained_model(model)

    transform = A.Compose(
        [
            A.RandomCrop(224, 224),
            A.HorizontalFlip(),
            A.augmentations.transforms.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
            # A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
            # A.RandomGamma(gamma_limit=(90, 110)),
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
            A.augmentations.transforms.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
            A.Normalize(),
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
    validate_dataset = ValidateDataset(transform=validate_transform, use_rate=0.3)
    train_class.set_train_dataloader(dataset=train_dataset)
    train_class.set_validation_dataloader(dataset=validate_dataset)
    train_class.enable_ckpt(f"models/ckpt/{model_name}")
    train_class.enable_tensorboard(save_image_log=True)
    train_class.train()
