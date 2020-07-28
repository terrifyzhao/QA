from sim_model.model import BIMPM
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sim_model.data_process import *
import torch


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

    train_data_set = TensorDataset(torch.from_numpy(train_p), torch.from_numpy(train_q), torch.from_numpy(train_y))
    train_data_loader = DataLoader(dataset=train_data_set, batch_size=batch_size, shuffle=True)

    eval_p = padding_seq(eval_p.apply(seq2index))
    eval_q = padding_seq(eval_q.apply(seq2index))
    return train_data_loader, [eval_p, eval_q], eval_y.values


# 训练模型
def train():
    train_data_loader, eval_x, eval_y = load_data(64)
    model = BIMPM(char_vocab_size=3966,
                  char_dim=300,
                  char_hidden_size=256,
                  hidden_size=256,
                  max_word_len=15)
    model.fit(train_data_loader, (eval_x, eval_y), 10, model_path='bimpm.p', lr=0.01)


if __name__ == '__main__':
    train()
