import torch.nn as nn


class ConvNS(nn.Module):
    def __init__(self):
        super(ConvNS, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class ConvNSExpr(nn.Module):
    def __init__(self):
        super(ConvNSExpr, self).__init__()

        self.act = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv0 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)

        self.adaptive = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128, 1024)
        self.linear2 = nn.Linear(1024, 82)

    def forward(self, x):
        out = self.conv0(x)
        out = self.act(out)
        out = self.pool(out)

        out = self.conv1(out)
        out = self.act(out)
        out = self.pool(out)

        out = self.conv2(out)
        out = self.act(out)
        out = self.pool(out)

        out = self.conv3(out)
        out = self.act(out)
        out = self.pool(out)

        out = self.adaptive(out)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)

        return out
