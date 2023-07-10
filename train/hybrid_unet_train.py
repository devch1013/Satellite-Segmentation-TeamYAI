from train.train_class import Trainer
from models.hybrid_unet import HybridUNet
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.dataloader import SatelliteDataset, validate_separator


if __name__ == "__main__":
    train_class = Trainer(base_dir="/root/dacon", config_dir="models/hybrid_unet_config.yaml")
    train_class.set_model(HybridUNet, state_dict="models/ckpt/hybridUnet_crop_29_07-10-14:40")
    transform = A.Compose([
        A.RandomCrop(224, 224),
        A.Normalize(),
        A.HorizontalFlip(), # Same with transforms.RandomHorizontalFlip()
        ToTensorV2()
    ])
    train_dataset, validate_dataset = validate_separator(csv_file='data/train.csv', transform=transform)
    train_class.set_train_dataloader(dataset=train_dataset)
    train_class.set_validation_dataloader(dataset=validate_dataset)
    train_class.enable_ckpt("models/ckpt")
    train_class.enable_tensorboard(save_image_log=True)
    train_class.train()
    
