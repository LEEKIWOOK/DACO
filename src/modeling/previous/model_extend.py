import torch
import torch.nn as nn

drop_rate = 0.3

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Flattening(nn.Module):
    def __init__(self):
        super(Flattening, self).__init__()
    
    def forward(self, x): 
        return torch.flatten(x, 1)

class BasicBlock3x3(nn.Module):
    expansion = 1
    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        
        self.conv_bn = nn.Sequential(
            conv3x3(inplanes3, planes, stride),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            conv3x3(planes, planes),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv_bn(x)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class Predictor(nn.Module):
    def __init__(self, input_channel, layers=[1, 1, 1, 1]):
        super(Predictor, self).__init__()
        self.inplanes3 = 100

        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 32, layers[0], stride=2)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 64, layers[0], stride=2)
        self.maxpool3 = nn.AvgPool1d(kernel_size=3, stride=1, padding=0)

        self.flattening = Flattening()
        self.fc = nn.Sequential(
            nn.Linear(960, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(32, 1)
        )

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
                nn.ReLU(),
                nn.Dropout(drop_rate),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def forward(self, v3, v4, v5):
        
        x = self.layer3x3_1(v3)
        x = self.maxpool3(x)
        x = self.layer3x3_2(x)
        x = self.maxpool3(x) #512, 64, 5

        y = self.layer3x3_1(v4)
        y = self.maxpool3(y)
        y = self.layer3x3_2(y)
        y = self.maxpool3(y) #512, 64, 5

        z = self.layer3x3_1(v5)
        z = self.maxpool3(z)
        z = self.layer3x3_2(z)
        z = self.maxpool3(z) #512, 64, 5

        out = torch.cat([x, y, z], dim=2)
        out = self.flattening(out)
        #print(out.shape)
        out = self.fc(out)
        return out.squeeze()