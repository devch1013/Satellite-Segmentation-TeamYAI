from train.train_class import Trainer
from models.MAResUnet.MAResUnet import MAResUNet
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.dataloader import SatelliteDataset, validate_separator

model_name = "MAResUnet_cont"
root_dir = "/root/dacon"


if __name__ == "__main__":
    train_class = Trainer(base_dir=root_dir, config_dir=f"models/{model_name}.yaml")
    train_class.set_model(
        MAResUNet,
        state_dict="/root/dacon/models/ckpt/MAResUnet/MAResUnet_msssim_541_07-24-16:40",
    )

    transform = A.Compose(
        [
            A.RandomCrop(224, 224),
            A.HorizontalFlip(),

            # A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
            # A.RandomGamma(gamma_limit=(90, 110)),
            A.augmentations.transforms.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
            # A.augmentations.transforms.ColorJitter(p=0.5),
            # A.augmentations.transforms.RandomShadow(
            #     shadow_roi=(0, 0, 1, 1),
            #     num_shadows_lower=10,
            #     num_shadows_upper=20,
            #     shadow_dimension=5,
            #     p=0.6,
            # ),
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
