import torch

# hyperparameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.01
bs = 32
n_classes = 10
epochs = 1
input_shape = 3 * 32 * 32
h1 = 1024
h2 = 512
train = True
