import torch
from torch import nn


class TextCNN(nn.Module):

    def __init__(self,
                 vocab_len,
                 embedding_size=100,
                 max_len=10):
        super().__init__()

        # 词典大小，embedding维度
        self.embedding = nn.Embedding(vocab_len, embedding_size)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(2, embedding_size))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3, embedding_size))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(4, embedding_size))

        # seq_len - kernel_size[0] + 1
        self.max_pool1 = nn.MaxPool1d(kernel_size=max_len - 2 + 1)
        self.max_pool2 = nn.MaxPool1d(kernel_size=max_len - 3 + 1)
        self.max_pool3 = nn.MaxPool1d(kernel_size=max_len - 4 + 1)

        self.dense = nn.Linear(6, 1)
        self.drop_out = nn.Dropout(0.2)

    # 前向传播
    def forward(self, x):
        # [batch_size, seq_len, embedding_size]
        embedding = self.embedding(x)
        # 加一个维度  [batch_size, 1, seq_len, embedding_size]
        embedding = embedding.unsqueeze(1)

        conv1_out = self.conv1(embedding).squeeze(-1)
        conv2_out = self.conv2(embedding).squeeze(-1)
        conv3_out = self.conv3(embedding).squeeze(-1)

        out1 = self.max_pool1(conv1_out)
        out2 = self.max_pool2(conv2_out)
        out3 = self.max_pool3(conv3_out)

        out = torch.cat([out1, out2, out3], dim=1).squeeze(-1)

        out = self.drop_out(out)
        out = self.dense(out)
        out = torch.sigmoid(out).squeeze(-1)
        return out
