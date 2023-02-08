# importing the libraries
import torch, torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import time
import torch.nn.functional as F
from tqdm import tqdm

# hyperparameter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.01
bs = 32
n_classes = 10
epochs = 10
input_shape = 3 * 32 * 32
h1 = 1024
h2 = 512
train = True

# For reproducibility
torch.manual_seed(42)

# importing the dataset
print('Downloading the dataset')
root = "/root/sonu/cv-experiments/"

# Transform
tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.49139968, 0.48215827 ,0.44653124], [0.24703233, 0.24348505 ,0.26158768])])

# Train and valid dataset
train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=tfms)
valid = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=tfms)
print('Shape of train datast is {} and Shape of valid dataset is {}'.format(len(train), len(valid)))

# Creating dataloader
train_dl = DataLoader(train, batch_size=bs, drop_last=True)
valid_dl = DataLoader(valid, batch_size=bs, shuffle=False, drop_last=True)

# Taking one batch of dataset
xb, yb = next(iter(train_dl))
print('shape of one batch of train dataloader', xb.shape, yb.shape)

# Model
class CifarModel(nn.Module):
    def __init__(self, num_class):
        super(CifarModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 1024),        # 3072 ----> 512
            nn.ReLU(),
            nn.Linear(1024, 512),        # 3072 ----> 512
            nn.ReLU(),
            nn.Linear(512, 124),        # 3072 ----> 512
            nn.ReLU(),
            nn.Linear(124, 64),             # 512 ------> 64
            nn.ReLU(),
            nn.Linear(64, num_class)        # 64 --------> 10
        )
    def forward(self, xb):
        xb = xb.view(bs, -1)           # (32, 3, 32, 32) ---> (32, 3*32*32)
        out = self.model(xb)
        return out

# Training loop
def train_model():
    tic = time.time()
    for i in range(epochs):
        running_loss = 0
        for xb, yb in tqdm(train_dl, desc='Training'):
            xb = xb.to(device)
            yb = yb.to(device)
            # forward
            logit = model(xb)
            loss = F.cross_entropy(logit, yb)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Running loss
            running_loss += loss.item()
        print('| Epoch :{0} | loss : {1:.4} |'.format(i, running_loss/len(train_dl)))
    tok = time.time()
    print('total time take to train is {:.2} s'.format(tok-tic))
    # Save Model
    torch.save(model.state_dict(), 'model.pt')
    return model

# Load Model
model = CifarModel(n_classes)
model.to(device)

# If train is true then train the model else load the saved model
if train:
    # Defining optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model = train_model()
else:
    try:
        model.load_state_dict(torch.load('model.pt'))
    except:
        # Defining optimizer and loss function
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        model = train_model()

# Compute loss on valid dataset
@torch.no_grad()
def check_accuracy(valid_dl):
    running_loss = 0
    model.eval()
    for xb, yb in tqdm(valid_dl, desc='Validation'):
        xb = xb.to(device)
        yb = yb.to(device)

        # forward
        logit = model(xb)
        loss = F.cross_entropy(logit, yb)

        running_loss += loss.item()

    print('| Valid loss is : {:.2} |'.format(running_loss/len(valid_dl)))

# Testing validation loss
check_accuracy(valid_dl)



