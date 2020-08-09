import sys
import os

rootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootPath)

from text_similarity.esim import ESIM
from torch.utils.data import DataLoader, TensorDataset
from text_similarity.data_process import *
import torch
from torch import nn


def load_data(batch_size=32):
    df = pd.read_csv('../data/LCQMC.csv')
    train_df, eval_df = split_data(df)
    train_p = df['sentence1']
    train_q = df['sentence2']
    train_y = df['label']
    eval_p = eval_df['sentence1']
    eval_q = eval_df['sentence2']
    eval_y = eval_df['label']

    train_p = padding_seq(train_p.apply(seq2index))
    train_q = padding_seq(train_q.apply(seq2index))
    train_y = np.array(train_y)

    train_data_set = TensorDataset(torch.from_numpy(train_p),
                                   torch.from_numpy(train_q),
                                   torch.from_numpy(train_y))
    train_data_loader = DataLoader(dataset=train_data_set, batch_size=batch_size, shuffle=True)

    eval_p = padding_seq(eval_p.apply(seq2index))
    eval_q = padding_seq(eval_q.apply(seq2index))
    return train_data_loader, [eval_p, eval_q], eval_y.values


# 训练模型
def train():
    model = ESIM(char_vocab_size=3966,
                 char_dim=100,
                 char_hidden_size=128,
                 hidden_size=128,
                 max_word_len=10)

    train_data_loader, eval_x, eval_y = load_data(512)
    eval_p = eval_x[0]
    eval_q = eval_x[1]
    eval_p = torch.from_numpy(eval_p)
    eval_q = torch.from_numpy(eval_q)
    if torch.cuda.is_available():
        model = model.cuda()
        eval_p = eval_p.cuda()
        eval_q = eval_q.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.BCELoss()

    best_acc = 0

    for epoch in range(10):
        for step, (b_p, b_q, b_y) in enumerate(train_data_loader):
            if torch.cuda.is_available():
                b_p = b_p.cuda()
                b_q = b_q.cuda()
                b_y = b_y.cuda()

            output = model(b_p, b_q)

            loss = loss_func(output, b_y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                test_output = model(eval_p, eval_q)
                pred_y = (test_output.cpu().data.numpy() > 0.5).astype(int)
                accuracy = float((pred_y == eval_y).astype(int).sum()) / float(eval_y.size)
                if accuracy > best_acc:
                    best_acc = accuracy
                    torch.save(model, 'esim.p')
                    print('save model, accuracy: %.3f' % accuracy)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(),
                      '| test accuracy: %.3f' % accuracy)


if __name__ == '__main__':
    train()
