import torch
import torch.nn as nn

drop_rate = 0.3

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride, padding=1, bias=False)

def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride, padding=1, bias=False)

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

class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()

        self.conv_bn = nn.Sequential(
            conv5x5(inplanes5, planes, stride),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            conv5x5(planes, planes),
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

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:,:,0:-d] + out
        out1 = self.relu(out1)
        return out1

class BasicBlock7x7(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock7x7, self).__init__()
        
        self.conv_bn = nn.Sequential(
            conv7x7(inplanes7, planes, stride),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            conv7x7(planes, planes),
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

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)
        return out1

class Predictor(nn.Module):
    def __init__(self, input_channel, layers=[1, 1, 1, 1]):
        super(Predictor, self).__init__()
        self.inplanes3 = 128
        self.inplanes5 = 128
        self.inplanes7 = 128

        self.pre_conv = nn.Sequential(
            nn.Conv1d(input_channel, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
        )

        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 64, layers[0], stride=2)
        self.maxpool3 = nn.AvgPool1d(kernel_size=3, stride=1, padding=0)

        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, 64, layers[0], stride=2)
        self.maxpool5 = nn.AvgPool1d(kernel_size=3, stride=1, padding=0)

        self.layer7x7_1 = self._make_layer7(BasicBlock7x7, 64, layers[0], stride=2)

        self.flattening = Flattening()
        self.fc = nn.Sequential(
            nn.Linear(704, 128),
            #nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(32, 1)
        )

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
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

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
                nn.ReLU(),
                nn.Dropout(drop_rate),
            )

        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)


    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
                nn.ReLU(),
                nn.Dropout(drop_rate),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def forward(self, e):
        
        x0 = self.pre_conv(e)

        x = self.layer3x3_1(x0)
        x = self.maxpool3(x)

        y = self.layer5x5_1(x0)
        y = self.maxpool5(y)

        z = self.layer7x7_1(x0)

        out = torch.cat([x, y, z], dim=2)
        out = self.flattening(out)
        #print(out.shape)
        out = self.fc(out)
        return out.squeeze()