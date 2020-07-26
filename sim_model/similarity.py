import torch.nn as nn
import torch
import torch.optim as optim
from elmoformanylangs import Embedder
import os
import pandas as pd
import numpy as np
import logging

logging.getLogger().setLevel(logging.ERROR)

elmo = Embedder(os.path.join(os.getcwd(), '../elmo'))


class SimModel(nn.Module):
    def __init__(self):
        super(SimModel, self).__init__()
        self.fc = nn.Linear(1024 * 2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x
        # 使用elmo的输出值作为输入
        # x1 = elmo.sents2elmo(x1)
        # x1 = torch.from_numpy(np.array([np.mean(v, axis=0) for v in x1])).cuda()
        # x2 = elmo.sents2elmo(x2)
        # x2 = torch.from_numpy(np.array([np.mean(v, axis=0) for v in x2])).cuda()
        # 拼接在一起
        x = torch.cat([x1, x2], dim=1)
        # 简单做一个二分类
        x = self.fc(x)
        return x


df = pd.read_csv('../data/LCQMC.csv')


def data_generator(batch_size):
    while True:
        sentence1 = []
        sentence2 = []
        labels = []
        for data in df.values:
            sentence1.append(data[0])
            sentence2.append(data[1])
            labels.append(torch.tensor(data[2], dtype=torch.long))

            if len(labels) == batch_size:
                sentence1 = elmo.sents2elmo(sentence1)
                sentence1 = [np.mean(v, axis=0) for v in sentence1]
                sentence2 = elmo.sents2elmo(sentence2)
                sentence2 = [np.mean(v, axis=0) for v in sentence2]
                yield [torch.from_numpy(np.array(sentence1)).cuda(), torch.from_numpy(np.array(sentence2)).cuda()], \
                      torch.from_numpy(np.array(labels)).cuda()
                sentence1, sentence2, labels = [], [], []


net = SimModel().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(data_generator(128), 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        pred = outputs.argmax(dim=1)
        num_correct = torch.eq(pred, labels).sum().float().item()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 0:  # print every 10 mini-batches
            print('[%d, %5d] loss: %.5f acc: %.4f' %
                  (epoch + 1, i + 1, running_loss / 10, num_correct / 128))
            running_loss = 0.0

print('Finished Training')
