import pandas as pd
from collections import defaultdict
import numpy as np
import os

path = os.path.dirname(__file__)


def tokenize(string):
    # res = list(jieba.cut(string, cut_all=False))
    res = list(string)
    return res


# 构建词典
def build_vocab(del_word_frequency):
    data = pd.read_csv('../data/LCQMC.csv')
    segment1 = data['sentence1'].apply(tokenize)
    segment2 = data['sentence2'].apply(tokenize)

    word_frequency = defaultdict(int)
    for row in segment1 + segment2:
        for i in row:
            word_frequency[i] += 1

    word_sort = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)  # 根据词频降序排序

    f = open('vocab.txt', 'w', encoding='utf-8')
    f.write('[PAD]' + "\n" + '[UNK]' + "\n")
    for d in word_sort:
        if d[1] > del_word_frequency:
            f.write(d[0] + "\n")
    f.close()


# 划分训练集和测试集
def split_data(df, split=0.7):
    df = df.sample(frac=1)
    length = len(df)
    train_data = df[0:length - 5000]
    eval_data = df[length - 5000:]

    return train_data, eval_data


vocab = {}
if os.path.exists(path + '/vocab.txt'):
    with open(path + '/vocab.txt', encoding='utf-8')as file:
        for line in file.readlines():
            vocab[line.strip()] = len(vocab)


# 把数据转换成index
def seq2index(seq):
    seg = tokenize(seq)
    seg_index = []
    for s in seg:
        seg_index.append(vocab.get(s, 1))
    return seg_index


# 统一长度
def padding_seq(X, max_len=10):
    return np.array([
        np.concatenate([x, [0] * (max_len - len(x))]) if len(x) < max_len else x[:max_len] for x in X
    ])


if __name__ == '__main__':
    build_vocab(5)
