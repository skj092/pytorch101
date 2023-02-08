from torch import nn
import config as cfg


# Model
class CifarModel(nn.Module):
    def __init__(self, num_class):
        super(CifarModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(cfg.input_shape, 1024),  # 3072 ----> 512
            nn.ReLU(),
            nn.Linear(1024, 512),  # 3072 ----> 512
            nn.ReLU(),
            nn.Linear(512, 124),  # 3072 ----> 512
            nn.ReLU(),
            nn.Linear(124, 64),  # 512 ------> 64
            nn.ReLU(),
            nn.Linear(64, num_class),  # 64 --------> 10
        )

    def forward(self, xb):
        xb = xb.view(cfg.bs, -1)  # (32, 3, 32, 32) ---> (32, 3*32*32)
        out = self.model(xb)
        return out
