from train.train_class_segdiff import TrainerSegdiff
from models.unet.UNet3Plus import UNet3Plus, UNet3Plus_DeepSup_CGM, UNet3Plus_DeepSup
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.dataloader import SatelliteDataset, validate_separator
from med_seg_diff_pytorch import Unet, MedSegDiff

model_name = "segdiff"
root_dir = "/root/dacon"


if __name__ == "__main__":
    train_class = TrainerSegdiff(base_dir=root_dir, config_dir=f"models/{model_name}.yaml")
    # train_class.set_model(
    #     UNet3Plus_DeepSup,
    #     # state_dict="/home/work/Dacon-YAI/devch/models/ckpt/unet3plus_deepsup_cgm/Unet3plus_deepsup_cgm_shallow_266_07-21-14:22",
    # )
    model = Unet(
        dim=64,
        image_size=224,
        mask_channels=1,  # segmentation has 1 channel
        input_img_channels=3,  # input images have 3 channels
        dim_mults=(1, 2, 4, 8),
    )

    diffusion = MedSegDiff(model, timesteps=100).cuda()
    train_class.set_pretrained_model(diffusion)
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
