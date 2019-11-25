'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1), # prelayer 192 output size of last layer
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 226, 242)
        self.b3 = Inception(660, 128, 128, 192, 32, 96, 64) #changed

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        # self.linear = nn.Linear(1024, 10)
        self.linear = nn.Linear(2764800, 1) #CHANGED

    def forward(self, x):
        out = self.pre_layers(x)
        print("prelayer shape", out.shape)
        out = self.a3(out)
        print("a3 shape", out.shape)
        out = self.b3(out)
        print("b3 shape", out.shape)
        out = self.maxpool(out)
        print("maxpool shape", out.shape)
        out = self.a4(out)
        print("a4 shape", out.shape)
        out = self.b4(out)
        print("b4 shape", out.shape)
        out = self.c4(out)
        print("c4 shape", out.shape)
        out = self.d4(out)
        print("d4 shape", out.shape)
        out = self.e4(out)
        print("e4 shape", out.shape)
        out = self.maxpool(out)
        print("maxpool shape", out.shape)
        out = self.a5(out)
        print("a5 shape", out.shape)
        out = self.b5(out)
        print("b5 shape", out.shape)
        out = self.avgpool(out)
        print("avgpool shape", out.shape)
        out = out.view(out.size(0), -1)
        print("final out shape", out.shape)
        out = self.linear(out)
        print("linear out shape", out.shape)
        return out


def test():
    net = GoogLeNet()
    x = torch.randn(1,3,226,242) #226, 242, 3
    y = net(x)
    print(y.size())


if __name__ == '__main__':
    model = GoogLeNet()
    #data =
    result = test()
    print("result", result)

    #iterate over batches, predict classes get loss
    #use backprop on loss to update gradients

# test()
