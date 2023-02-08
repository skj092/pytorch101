# importing the libraries
import torch, torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import time
import torch.nn.functional as F
from tqdm import tqdm
import config as cfg
import engine, models


if __name__ == "__main__":
    # For reproducibility
    torch.manual_seed(42)

    # importing the dataset
    print("Downloading the dataset")
    root = "/root/sonu/cv-experiments/"

    # Transform
    tfms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.49139968, 0.48215827, 0.44653124],
                [0.24703233, 0.24348505, 0.26158768],
            ),
        ]
    )

    # Train and valid dataset
    train = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=tfms
    )
    valid = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=tfms
    )
    print(
        "Shape of train datast is {} and Shape of valid dataset is {}".format(
            len(train), len(valid)
        )
    )

    # Creating dataloader
    train_dl = DataLoader(train, batch_size=cfg.bs, drop_last=True)
    valid_dl = DataLoader(valid, batch_size=cfg.bs, shuffle=False, drop_last=True)

    # Taking one batch of dataset
    xb, yb = next(iter(train_dl))
    print("shape of one batch of train dataloader", xb.shape, yb.shape)

    # Load Model
    model = models.CifarModel(cfg.n_classes)
    model.to(cfg.device)

    # If train is true then train the model else load the saved model
    if train:
        # Defining optimizer and loss function
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)
        model = engine.train_model(
            model, train_dl=train_dl, valid_dl=valid_dl, optimizer=optimizer
        )
    else:
        try:
            model.load_state_dict(torch.load("model.pt"))
        except:
            # Defining optimizer and loss function
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)
            model = engine.train_model(
                model, train_dl=train_dl, valid_dl=valid_dl, optimizer=optimizer
            )

    torch.save(model.state_dict(), "model.pt")
