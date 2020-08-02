import torch
from text_similarity.data_process import seq2index, padding_seq
import os

path = os.path.dirname(__file__)
model = torch.load(path + '/esim.p')
model.eval()


def predict(p, q):
    p = [seq2index(i) for i in p]
    q = [seq2index(i) for i in q]
    p = torch.from_numpy(padding_seq(p)).cuda()
    q = torch.from_numpy(padding_seq(q)).cuda()
    out = model(p, q)
    return out.cpu().data.numpy()


if __name__ == '__main__':
    print(predict(['投了安邦长青树，给的却是电子保单，这个有效力吗？'], ['投的安邦长青树是电子保单，有效吗']))
