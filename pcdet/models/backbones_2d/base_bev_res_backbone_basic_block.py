import torch

class BaseBEVResBackboneBasicBlock(torch.nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            padding: int = 1,
            downsample: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu2 = torch.nn.ReLU()
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = torch.nn.Sequential(
                torch.nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                torch.nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_layer(x)

        out += identity
        out = self.relu2(out)

        return out
