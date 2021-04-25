import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import resnet50


class Model(nn.Module):
    FEAT_SIZE = 1000
    INTERMEDIATE_FEAT_SIZE = 128
    DROP_RATE = 0.6
    NUM_CONTROLS = 2
    def __init__(self):
        super(Model, self).__init__()
        self.feat = resnet50(pretrained=True)
        self.control = nn.Sequential(
            nn.Dropout(p=self.DROP_RATE),
            nn.Linear(in_features=self.FEAT_SIZE, out_features=self.INTERMEDIATE_FEAT_SIZE, bias=True),
            nn.Dropout(p=self.DROP_RATE),
            nn.LeakyReLU(inplace=True, negative_slope=0.3),
            nn.Linear(in_features=self.INTERMEDIATE_FEAT_SIZE, out_features=self.NUM_CONTROLS, bias=True)
        )
        print(f'ResNet control model. Dropout rate {self.DROP_RATE}, '
              f'intermediate fc size {self.INTERMEDIATE_FEAT_SIZE}')

    def forward(self, x):
        x = self.feat(x)
        x = self.control(x)

        return x


if __name__ == "__main__":
    model = Model()
    x = torch.randn(2, 3, 112, 112)
    print(model(x).shape)
