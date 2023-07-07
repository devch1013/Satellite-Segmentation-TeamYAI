import torch
import yaml
import sys
from tqdm.auto import tqdm
import os
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import datetime
from pathlib import Path
from utils.utils import *
from utils.dataloader import MyDataLoader
from models.layers import VIT


class Trainer:
    def __init__(self, base_dir: str, config_dir: str = "models/config.yaml"):
        self.save_ckpt = False
        self.base_dir = base_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = self._load_config(config_dir)
        self.model = None
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.writer = None

    def _load_config(self, config_dir: str):
        with open(config_dir) as f:
            cfg = yaml.safe_load(f)
        return cfg

    def enable_ckpt(self, save_dir: str):
        assert save_dir is not None and self.base_dir is not None
        self.save_ckpt = True
        self.save_dir = save_dir

    def enable_tensorboard(self, log_dir: str = None):
        current_time = datetime.datetime.now() + datetime.timedelta(hours=9)
        current_time = current_time.strftime("%m-%d-%H:%M")
        Path(os.path.join(self.base_dir, f"log")).mkdir(parents=True, exist_ok=True)
        if log_dir is None:
            log_dir = f"log/{self.cfg['model-name']}__epochs-{self.cfg['train']['epoch']}_batch_size-{self.cfg['train']['batch-size']}__{current_time}"
        tensorboard_dir = os.path.join(self.base_dir, log_dir)
        self.writer = SummaryWriter(tensorboard_dir)

    def set_model(self, model_class):
        '''
        
        '''
        self.model = model_class(self.cfg["model"], self.device).to(self.device)
        self.optimizer = get_optimizer(self.model, self.cfg["train"]["optimizer"])
        self.criterion = get_criterion(self.cfg["train"]["criterion"])

    def set_dataset(
        self,
        dataset_name: str =None,
        dataset=None,
        transform=None,
        dataset_root: str = "/home/ubuntu/datasets/",
        val=False,
    ):
        assert dataset_name != None or dataset != None
        if dataset == None:
            dataloader_class = MyDataLoader(
                dataset_name=dataset_name,
                dataset_root=dataset_root,
                transform=transform,
            )
            if val:
                (
                    self.train_dataset,
                    self.val_dataset,
                    self.test_dataset,
                ) = dataloader_class.get_dataloader_with_validation(self.cfg["train"]["batch-size"])
            else:
                self.train_dataset, self.test_dataset = dataloader_class.get_dataloader(
                    self.cfg["train"]["batch-size"]
                )
        else:
            self.train_dataset = DataLoader(dataset, batch_size=self.cfg["train"]["batch-size"], shuffle=True, num_workers=4)
            


    def train(
        self,
        epoch: int = 10,
    ):
        assert self.model is not None
        assert self.train_dataset is not None

        for i in range(epoch):
            self._train(i)
            # self._validate(i)

            # test(model, device, test_loader, criterion)

            # if batch_idx % test_interval == test_interval - 1 or batch_idx == len(train_loader) - 1:
            #     test(model, device, test_loader, criterion, args)
            

        if self.writer is not None:
            self.writer.close()

    def _train(self, current_epoch):
        self.model.train()
        total_loss = 0
        batch_idx = 0
        for data, target in tqdm(self.train_dataset):
            # print('*', end="")
            
            data, target = data.to(self.device), target.to(self.device, dtype=torch.long)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            if self.writer is not None:
                self.writer.add_scalar(
                    "Train Loss",
                    loss.item(),
                    batch_idx + current_epoch * (len(self.train_dataset)),
                )
            batch_idx +=1 
        if self.writer is not None:
            self.writer.add_scalar(
                "Train Loss per Epoch", total_loss / len(self.train_dataset), current_epoch
            )

    def inference(self, x):
        self.model.eval()
        self.model(x)

    def test(model, device, test_loader, criterion, args=None):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )

    def _validate(self, current_epoch):
        if self.val_dataset is None:
            return
        self.model.eval()
        validation_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_dataset:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                validation_loss += self.criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        validation_loss /= len(self.test_dataset)
        if self.writer is not None:
            self.writer.add_scalar("Validation Loss", validation_loss, current_epoch)
            self.writer.add_scalar(
                "Validation Accuracy",
                100.0 * correct / len(self.test_dataset.dataset),
                current_epoch,
            )


if __name__ == "__main__":
    train_class = Trainer(base_dir="/home/ubuntu/yaiconvally/VIT")
    train_class.set_model(VIT)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_class.set_dataset("cifar10", transform=transform, val=True)
    train_class.enable_tensorboard()
    train_class.train(epoch=10)
