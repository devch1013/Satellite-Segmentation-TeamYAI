import torch
import sys

sys.path.append("/home/ubuntu/yaiconvally")
from models.layers import VIT
from utils.dataloader import MyDataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import argparse
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter


def train(
    epoch,
    model,
    device,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    log_interval=10,
    test_interval=100,
    args=None,
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

        # if batch_idx % test_interval == test_interval - 1 or batch_idx == len(train_loader) - 1:
        #     test(model, device, test_loader, criterion, args)


def test(model, device, test_loader, criterion, args=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model training program")

    parser.add_argument("--img_size", default=32, type=int)
    parser.add_argument("--patch_size", default=2, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--epoch", default=30, type=int)
    parser.add_argument("--dataset", default="cifar10", type=str)

    args = parser.parse_args()
    epoch = args.epoch
    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = VIT(
        batch_size=batch_size,
        image_size=args.img_size,
        patch_size=args.patch_size,
        heads=4,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataclass = MyDataLoader(dataset_name="cifar10", transform=transform)

    train_loader, test_loader = dataclass.get_dataloader(batch_size=batch_size)
    for i in range(epoch):
        train(i, model, device, train_loader, test_loader, optimizer, criterion)
        test(model, device, test_loader, criterion)
