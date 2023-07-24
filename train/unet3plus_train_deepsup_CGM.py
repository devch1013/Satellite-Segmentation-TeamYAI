from train.train_class import Trainer
from models.unet.UNet3Plus import UNet3Plus, UNet3Plus_DeepSup_CGM
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.dataloader import SatelliteDataset, validate_separator

logfile_name = "unet3plus_deepsup_CGM"

if __name__ == "__main__":
    train_class = Trainer(base_dir="/root/dacon", config_dir=f"models/{logfile_name}.yaml")
    train_class.set_model(UNet3Plus_DeepSup_CGM)
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
    train_class.enable_ckpt(f"models/ckpt/{logfile_name}")
    train_class.enable_tensorboard(save_image_log=True)
    train_class.train()
