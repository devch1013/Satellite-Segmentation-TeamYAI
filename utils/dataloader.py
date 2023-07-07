from torchvision import datasets
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
from utils.utils import rle_decode


class MyDataLoader:
    def __init__(
        self,
        dataset_name="cifar100",
        transform=None,
        dataset_root: str = "/home/ubuntu/datasets/",
    ):
        self.dataset_root = dataset_root
        dataset_name = dataset_name.upper()
        if dataset_name == "CIFAR100":
            self.train_dataset, self.test_dataset = self._get_dataset(
                datasets.CIFAR100, dataset_name=dataset_name, transform=transform
            )
        elif dataset_name == "CIFAR10":
            self.train_dataset, self.test_dataset = self._get_dataset(
                datasets.CIFAR10, dataset_name=dataset_name, transform=transform
            )
        else:
            AssertionError("dataset_name is not supported")

    def _get_dataset(self, dataset_class, dataset_name, transform=None):
        train_dataset = dataset_class(
            root=self.dataset_root + dataset_name,
            train=True,
            download=True,
            transform=transform,
        )

        test_dataset = dataset_class(
            root=self.dataset_root + dataset_name,
            train=False,
            download=True,
            transform=transform,
        )

        return train_dataset, test_dataset

    def _get_shuffle_loader(self, dataset, batch_size):
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

    def get_train_loader(self, batch_size):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

    def get_test_loader(self, batch_size):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )

    def get_dataloader(self, batch_size):
        return self.get_train_loader(batch_size), self.get_test_loader(batch_size)

    def get_dataloader_with_validation(self, batch_size):
        testloader = self.get_test_loader(batch_size)
        train_dataset, validation_data = torch.utils.data.random_split(
            self.train_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(1)
        )
        trainloader = self._get_shuffle_loader(train_dataset, batch_size)
        validationloader = self._get_shuffle_loader(validation_data, batch_size)
        return trainloader, validationloader, testloader


class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread("data"+img_path[1:])
        # print(img_path.replace(".","data"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask
    
    
    
if __name__ == "__main__":
    image = cv2.imread("data/train_img/TRAIN_4390.png")
    print(image)