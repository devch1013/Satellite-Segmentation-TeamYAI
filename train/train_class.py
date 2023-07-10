import torch
import yaml
import sys
from tqdm.auto import tqdm
import os
import torchvision.transforms as transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import datetime
from pathlib import Path
from utils.utils import *
from utils.loss_func import dice_loss
from utils.dataloader import MyDataLoader
from models.layers import VIT
import torch.nn.functional as F


class Trainer:
    def __init__(self, base_dir: str, config_dir: str = "models/config.yaml"):
        self.save_ckpt = False
        self.base_dir = base_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = self._load_config(config_dir)
        self.model = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.val_dataloader = None
        self.validation = False
        self.writer = None
        

    def _load_config(self, config_dir: str):
        with open(config_dir) as f:
            cfg = yaml.safe_load(f)
        return cfg

    def enable_ckpt(self, save_dir: str):
        assert save_dir is not None and self.base_dir is not None
        self.save_ckpt = True
        self.save_dir = save_dir

    def enable_tensorboard(self, log_dir: str = None, save_image_log=False):
        self.save_image_log = save_image_log
        print(self.save_image_log)
        current_time = datetime.datetime.now() + datetime.timedelta(hours=9)
        current_time = current_time.strftime("%m-%d-%H:%M")
        Path(os.path.join(self.base_dir, f"log")).mkdir(parents=True, exist_ok=True)
        if log_dir is None:
            log_dir = f"log/{self.cfg['model-name']}__epochs-{self.cfg['train']['epoch']}_batch_size-{self.cfg['train']['batch-size']}__{current_time}"
        tensorboard_dir = os.path.join(self.base_dir, log_dir)
        self.writer = SummaryWriter(tensorboard_dir)

    def set_model(self, model_class, state_dict=None):
        '''
        
        '''
        self.model = model_class(self.cfg["model"], self.device).to(self.device)
        if state_dict != None:
            self.model.load_state_dict(torch.load(state_dict))
        self.optimizer = get_optimizer(self.model, self.cfg["train"]["optimizer"])
        self.criterion = get_criterion(self.cfg["train"]["criterion"])
        self.scheduler = get_scheduler(self.optimizer, self.cfg["train"]["lr_scheduler"])

    def set_train_dataloader(
        self,
        dataset,
    ):
        print("data loader setting complete")
        self.train_dataloader = DataLoader(dataset, batch_size=self.cfg["train"]["batch-size"], shuffle=True, num_workers=16)
        
    def set_validation_dataloader(
        self,
        dataset,
    ):
        self.val_dataloader = DataLoader(dataset, batch_size=self.cfg["train"]["batch-size"], shuffle=True, num_workers=16)
        self.validation = True
            


    def train(
        self,
        epoch: int = None,
    ):
        assert self.model is not None
        assert self.train_dataloader is not None
        
        print("batch size: ", self.cfg["train"]["batch-size"])
        
        if epoch == None:
            epoch = self.cfg["train"]["epoch"]

        for i in range(epoch):
            self._train(i)
            if self.validation:
                self._validate(i)

            # test(model, device, test_loader, criterion)

            # if batch_idx % test_interval == test_interval - 1 or batch_idx == len(train_loader) - 1:
            #     test(model, device, test_loader, criterion, args)
            

        if self.writer is not None:
            self.writer.close()

    def _train(self, current_epoch):
        self.model.train()
        total_loss = 0
        batch_idx = 0
        for data, target in tqdm(self.train_dataloader):
            # print('*', end="")
            
            data, target = data.to(self.device, dtype=torch.float32), target.to(self.device, dtype=torch.float32)
            self.optimizer.zero_grad()
            output = self.model(data)
            output = output.squeeze(dim=1)
            loss = self.criterion(output, target)
            dice = dice_loss(
                F.softmax(output, dim=1).float(),
                target,
                multiclass=False
            )
            loss += dice
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            if self.writer is not None:
                self.writer.add_scalar(
                    "Train Loss",
                    loss.item(),
                    batch_idx + current_epoch * (len(self.train_dataloader)),
                )
                self.writer.add_scalar(
                    "Dice score",
                    (dice-1)*-1,
                    batch_idx + current_epoch * (len(self.train_dataloader)),
                )
                self.writer.add_scalar(
                "learning rate", self.optimizer.param_groups[0]["lr"], current_epoch
                )
            batch_idx +=1 
        if self.validation:
            self.model.eval()
            with torch.no_grad():
                loss = self.criterion(output, target)
                # print(target.size())
                dice = dice_loss(
                    F.softmax(output, dim=1).float(),
                    target,
                    multiclass=False
                )
                loss += dice
                
        self.scheduler.step()
        if self.writer is not None:
            self.writer.add_scalar(
                "Train Loss per Epoch", total_loss / len(self.train_dataloader), current_epoch
            )
            
            if self.save_image_log:
                origin_img = torchvision.utils.make_grid(data[0])
                data_img = torchvision.utils.make_grid((output[0] > 0.35).to(dtype=torch.int32))
                mask_img = torchvision.utils.make_grid(target[0])
                self.writer.add_image('inter_result_origin', origin_img, current_epoch)
                self.writer.add_image('inter_result_output', data_img, current_epoch)
                self.writer.add_image('inter_result_mask', mask_img, current_epoch)
        if self.save_ckpt:
            current_time = datetime.datetime.now() + datetime.timedelta(hours=9)
            current_time = current_time.strftime("%m-%d-%H:%M")
            model_name = self.cfg["model-name"]
            torch.save(self.model.state_dict(), f"{self.save_dir}/{model_name}_{current_epoch}_{current_time}")

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
