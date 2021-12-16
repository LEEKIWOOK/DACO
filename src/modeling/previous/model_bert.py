import torch
import torch.nn as nn
from modeling.utils import Flattening

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        
        dropout_rate = 0.2
        self.embedding_dim = 768
        self.out_dim = 64
        self.seq_len = 31
        
        self.dropout = nn.Dropout(dropout_rate)
        self.flattening = Flattening()
        
        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.out_dim, kernel_size = 3, stride=1, padding="same"),
            nn.BatchNorm1d(self.out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(kernel_size = 2, stride = 1, padding=1)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.out_dim, kernel_size = 5, stride=1, padding="same"),
            nn.BatchNorm1d(self.out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(kernel_size = 2, stride = 1, padding=1)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.out_dim, kernel_size = 7, stride=1, padding="same"),
            nn.BatchNorm1d(self.out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(kernel_size = 2, stride = 1, padding=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.out_dim * 3 * self.seq_len, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )
        

    def forward(self, e):
        
        #e = self.bert(e)
        x0 = e.permute(0, 2, 1)

        x1 = self.conv_layer1(x0)
        x2 = self.conv_layer2(x0)
        x3 = self.conv_layer3(x0)

        output = torch.cat((x1, x2, x3), dim = 1)
        output = output.permute(0, 2, 1)
        output = self.dropout(output)
        output = self.flattening(output)
        
        output = self.fc(output)
        return output.squeeze()
        