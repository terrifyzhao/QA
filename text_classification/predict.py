import torch
from text_classification.data_process import seq2index, padding_seq
import os

path = os.path.dirname(__file__)

model = torch.load(path + '/text_cnn.p')
model.eval()


def classification_predict(s):
    s = seq2index(s)
    s = torch.from_numpy(padding_seq([s])).cuda().long()
    out = model(s)
    return out.cpu().data.numpy()


if __name__ == '__main__':
    while 1:
        s = input('句子：')
        print(classification_predict(s))
