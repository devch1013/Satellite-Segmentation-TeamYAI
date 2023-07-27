from train.train_class_mask2mask import Trainer
from models.unet.mask2mask import Mask2Mask
from models.MAResUnet.MAResUnet import MAResUNet
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.dataloader import SatelliteDataset, validate_separator
import torch

model_name = "MAResUnet_mask2mask"
root_dir = "/root/dacon"

device = "cuda"

if __name__ == "__main__":
    train_class = Trainer(base_dir=root_dir, config_dir=f"models/{model_name}.yaml")
    cfg = train_class.cfg
    main_model = MAResUNet(**cfg["model"])
    filename = "/root/dacon/models/ckpt/MAResUnet/MAResUnet_msssim_541_07-24-16:40"
    main_model.load_state_dict(torch.load(filename))

    model = Mask2Mask(main_model).to(device)

    train_class.set_pretrained_model(model)

    transform = A.Compose(
        [
            A.RandomCrop(224, 224),
            A.HorizontalFlip(),
            A.augmentations.transforms.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
            # A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.4), contrast_limit=(-0.4, 0.4)),
            # A.RandomGamma(gamma_limit=(90, 110)),
            A.RandomRotate90(p=0.7),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    validate_transform = A.Compose(
        [
            A.RandomCrop(224, 224),
            A.Normalize(),
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
    train_class.set_train_dataloader(dataset=train_dataset)
    train_class.set_validation_dataloader(dataset=validate_dataset)
    train_class.enable_ckpt(f"models/ckpt/{model_name}")
    train_class.enable_tensorboard(save_image_log=True)
    train_class.train()
