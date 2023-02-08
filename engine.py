import torch.nn.functional as F
import config as cfg
import torch
import time
from tqdm import tqdm


# Compute loss on valid dataset
@torch.no_grad()
def check_accuracy(model, valid_dl):
    valid_loss = 0
    model.eval()
    for xb, yb in tqdm(valid_dl, desc="Validation"):
        xb = xb.to(cfg.device)
        yb = yb.to(cfg.device)

        # forward
        logit = model(xb)
        loss = F.cross_entropy(logit, yb)

        valid_loss += loss.item()

    return valid_loss / len(valid_dl)


# Training loop
def train_model(model, train_dl, valid_dl, optimizer):
    model.train()
    tic = time.time()
    for i in range(cfg.epochs):
        train_loss, valid_loss = 0, 0
        # Training
        for xb, yb in tqdm(train_dl, desc="Training"):
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            # forward
            logit = model(xb)
            loss = F.cross_entropy(logit, yb)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Running loss
            train_loss += loss.item()

        # Validation
        valid_loss = check_accuracy(model, valid_dl)

        print(
            "| Epoch :{0} | Train loss : {1:.4} | Valid loss = {2:.4} ".format(
                i, train_loss / len(train_dl), valid_loss
            )
        )
    tok = time.time()
    print("total time take to train is {:.2} s".format(tok - tic))

    return model
