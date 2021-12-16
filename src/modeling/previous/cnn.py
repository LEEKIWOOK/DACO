import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.utils import PositionalEncoding, Flattening


class CNN(nn.Module):
    def __init__(self, len: int):
        super(CNN, self).__init__()

        self.seq_len = len
        self.embedding_dim = 64
        self.dropout_rate = 0.3

        self.embedding_layer = nn.Embedding(
            num_embeddings=4, embedding_dim=self.embedding_dim, max_norm=True
        )
        self.position_encoding = PositionalEncoding(
            dim=self.embedding_dim, max_len=self.seq_len, dropout=0.1
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)

        self.ConvLayer = nn.Sequential(
            nn.Conv1d(
                self.embedding_dim,
                int(self.embedding_dim * 1.5),
                kernel_size=1,
                padding="same",
                stride=1,
                # bias=False,
            ),
            nn.BatchNorm1d(int(self.embedding_dim * 1.5)),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv1d(
                int(self.embedding_dim * 1.5),
                self.embedding_dim * 2,
                kernel_size=3,
                padding="same",
                stride=1,
                # bias=False,
            ),
            nn.BatchNorm1d(self.embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv1d(
                self.embedding_dim * 2,
                self.embedding_dim * 2,
                kernel_size=3,
                padding="same",
                stride=1,
                # bias=False,
            ),
            nn.BatchNorm1d(self.embedding_dim * 2),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )
        self.flattening = Flattening()

        self.predictor = nn.Sequential(
            nn.BatchNorm1d(3102),
            nn.Linear(in_features=3102, out_features=512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(512),
            nn.Linear(in_features=512, out_features=64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=1),
        )

    def attention(self, query, key, value, mask=None, dropout=0.0):
        # based on: https://nlp.seas.harvard.edu/2018/04/03/attention.html
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = F.dropout(p_attn, p=dropout, training=self.training)
        return torch.matmul(p_attn, value), p_attn

    def mh_attention(self, inputs, in_features):
        single_head_size = 16
        multi_head_num = 4
        multi_head_size = 64  ###

        Q = nn.ModuleList(
            [
                nn.Linear(
                    in_features=in_features,
                    out_features=single_head_size,
                )
                for i in range(0, multi_head_num)
            ]
        ).to(inputs.device)

        K = nn.ModuleList(
            [
                nn.Linear(
                    in_features=in_features,
                    out_features=single_head_size,
                )
                for i in range(0, multi_head_num)
            ]
        ).to(inputs.device)

        V = nn.ModuleList(
            [
                nn.Linear(
                    in_features=in_features,
                    out_features=single_head_size,
                )
                for i in range(0, multi_head_num)
            ]
        ).to(inputs.device)

        MultiHeadLinear = nn.Sequential(
            nn.LayerNorm(single_head_size * multi_head_num),
            nn.Linear(
                in_features=single_head_size * multi_head_num,
                out_features=multi_head_size,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        ).to(inputs.device)

        pAttn_concat = torch.Tensor([]).to(inputs.device)
        attn_concat = torch.Tensor([]).to(inputs.device)
        relu_list = nn.ModuleList([nn.ReLU() for i in range(0, multi_head_num)])

        for i in range(0, multi_head_num):
            query = Q[i](inputs)
            key = K[i](inputs)
            value = V[i](inputs)
            attnOut, p_attn = self.attention(
                query, key, value, dropout=self.dropout_rate
            )
            attnOut = relu_list[i](attnOut)
            attn_concat = torch.cat((attn_concat, attnOut), dim=2)
        attn_out = MultiHeadLinear(attn_concat)

        return attn_out

    def forward(self, inputs, embd):

        #embd = self.embedding_layer(inputs)
        #embd_pos = embd.clone()
        #embd_pos = embd_pos * math.sqrt(self.embedding_dim)
        #embd_pos = self.dropout(self.position_encoding(embd_pos))
        

        embd = embd.transpose(1, 2)
        embd = self.ConvLayer(embd)
        embd = self.flattening(embd)

        embd_pos = self.dropout(self.maxpool(self.mh_attention(embd_pos, 64)))
        embd_pos = self.flattening(embd_pos)

        # embd_cnn_F = self.flattening(self.mh_attention(embd_cnn, 16))  # [256, 256, 64]
        # embd_cnn_R = embd_cnn.transpose(1, 2)
        # embd_cnn_R = self.flattening(
        #    self.mh_attention(embd_cnn_R, 256)
        # )  # [256, 64, 256]

        # embd_F = self.flattening(self.mh_attention(embd_pos, 128))
        # embd_R = embd_pos.transpose(1, 2)
        # embd_R = self.flattening(self.mh_attention(embd_R, 33))

        output = torch.cat([embd, embd_pos], dim=1)
        output = self.predictor(output)
        return output.squeeze()
