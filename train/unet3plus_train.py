from train.train_class import Trainer
from models.unet.UNet3Plus import UNet3Plus
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.dataloader import SatelliteDataset, validate_separator


if __name__ == "__main__":
    train_class = Trainer(base_dir="/root/dacon", config_dir="models/unet3plus_1024channel.yaml")
    train_class.set_model(UNet3Plus)
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
        csv_file="data/train.csv", transform=transform, validation_ratio=0.9
    )
    train_class.set_train_dataloader(dataset=train_dataset)
    train_class.set_validation_dataloader(dataset=validate_dataset)
    train_class.enable_ckpt("models/ckpt/unet3plus_1024channel")
    train_class.enable_tensorboard(save_image_log=True)
    train_class.train()
