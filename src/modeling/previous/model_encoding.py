import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.utils import PositionalEncoding, Flattening


class CNN_GRU_ENC(nn.Module):
    def __init__(self, len: int):
        super(CNN_GRU_ENC, self).__init__()

        self.dropout_rate = 0.2
        self.rnn_hidden = 3 ### -> 3?

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.gru = nn.GRU(33, self.rnn_hidden, bidirectional=True)
        self.flattening = Flattening()
        self.dropout = nn.Dropout(self.dropout_rate)

        ##########################################################################

        self.ConvLayer_embd = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding="same", stride=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv1d(32, 64, kernel_size=3, padding="same", stride=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )

        self.predictor = nn.Sequential(
            nn.Linear(in_features=128, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features=32, out_features=1),
        )

        self.initialize_weights()
    
    def initialize_weights(self):
        # track all layers
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, inputs):

        reg_out = F.one_hot(inputs).to(torch.float)
        reg_out = reg_out.transpose(1, 2)
        reg_out = self.ConvLayer_embd(reg_out)

        reg_out, _ = self.gru(reg_out)
        F_RNN = reg_out[:, :, : self.rnn_hidden]
        R_RNN = reg_out[:, :, self.rnn_hidden :]
        reg_out = self.dropout(torch.cat((F_RNN, R_RNN), 2))
        
        reg_out = self.maxpool(reg_out)
        reg_out = self.flattening(reg_out) #256, 1023
        #print(reg_out.shape)
        reg_out = self.predictor(reg_out)

        return reg_out.squeeze()

        """
            Case 1. 
                (1)PE + Conv + FClayer
                (2)Kmer_embd + Conv + FClayer
                => epoch 40, corr : 0.4813269400504535
            Case 2.
                (1)reg : Multi-head attention
                (2)cls: Conv + gru + (reg) + FClayer
                => epoch 50, corr : 0.5083472740061205
            Case 3.
                (1)reg : Conv + Multi-head attention
                (2)cls: Conv + (reg) + FClayer
                => epoch 25, corr : 0.45469856093707905
            Case 4.
                (1)reg : Conv - embd matrix : 0.828542092506067 (epoch 500)
            Case 5.
                (1)reg : Conv - one hot encoding : 0.8097524308625214 (epoch 500)
            Case 6.
                (1)reg : Conv - embedding : 0.7057128779264171 (epoch 500)
            Case 7.
                (1)reg : Conv - embedding + GRU : 0.8288423534381729 (epoch 246)
            Case 8.
                (1)reg : Conv - embd matrix + mhsa + GRU  : 0.8061655301084629 (epoch 405)
            Case 9.
                (1)reg : embd matrix(pos) + mhsa + GRU  : 0
            Case 7 - 2 (GRU parameter tuning).
                (1)reg : Conv - embedding + GRU : 0.832914779724453

        """