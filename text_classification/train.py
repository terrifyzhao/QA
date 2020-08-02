import sys
import os

rootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootPath)

from text_classification.text_cnn import TextCNN
from torch.utils.data import DataLoader, TensorDataset
from text_classification.data_process import *
import torch
from torch import nn
import pandas as pd


def load_data(batch_size=32):
    df = pd.read_csv('../data/classification.csv')
    train_df, eval_df = split_data(df)
    train_x = df['sentence']
    train_y = df['label']
    eval_x = eval_df['sentence']
    eval_y = eval_df['label']

    train_x = padding_seq(train_x.apply(seq2index))
    train_y = np.array(train_y)

    train_data_set = TensorDataset(torch.from_numpy(train_x),
                                   torch.from_numpy(train_y))
    train_data_loader = DataLoader(dataset=train_data_set, batch_size=batch_size, shuffle=True)

    eval_x = padding_seq(eval_x.apply(seq2index))
    return train_data_loader, eval_x, eval_y.values


# 训练模型
def train():
    model = TextCNN(vocab_len=3966,
                    embedding_size=100)

    train_data_loader, eval_x, eval_y = load_data(512)

    eval_x = torch.from_numpy(eval_x)
    if torch.cuda.is_available():
        model = model.cuda()
        eval_x = eval_x.cuda().long()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.BCELoss()

    best_acc = 0

    for epoch in range(10):
        for step, (b_x, b_y) in enumerate(train_data_loader):
            if torch.cuda.is_available():
                b_x = b_x.cuda().long()
                b_y = b_y.cuda()
            output = model(b_x)
            loss = loss_func(output, b_y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                test_output = model(eval_x)
                pred_y = (test_output.cpu().data.numpy() > 0.5).astype(int)
                accuracy = float((pred_y == eval_y).astype(int).sum()) / float(eval_y.size)
                if accuracy > best_acc:
                    best_acc = accuracy
                    torch.save(model, 'text_cnn.p')
                    print('save model, accuracy: %.3f' % accuracy)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(),
                      '| test accuracy: %.3f' % accuracy)


if __name__ == '__main__':
    train()
