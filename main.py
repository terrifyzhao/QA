import pandas as pd
import gensim.models
import jieba
import numpy as np
import argparse
from elmoformanylangs import Embedder
import os
import joblib

# 加载数据
df = pd.read_csv('data/qa_data.csv')

question = df['question'].values
answer = df['answer'].values


def sen2vec(model, sentence, predict=False):
    """
    使用word2vec生成句向量
    :param model: word2vec的模型
    :param sentence: 句子
    :param predict: 是否是预测 如果是则会打印分词结果
    :return: 句向量
    """
    vec = np.zeros(100)
    segment = list(jieba.cut(sentence))
    if predict:
        print(segment)
    for s in segment:
        # 避免oov问题
        try:
            # 取词向量并相加
            vec += model.wv[s]
        except:
            pass
    # 计算均值得到句向量
    vec = vec / len(segment)
    return vec


def elmo2vec(model, sentence, predict=False):
    """
    使用elmo生成句向量
    :param model: elmo模型
    :param sentence: 句子
    :param predict: 是否是预测 如果是则会打印分词结果
    :return: 句向量
    """
    if isinstance(sentence, np.ndarray):
        segment = [jieba.cut(s) for s in sentence]
        vec = model.sents2elmo(segment)
    else:
        segment = list(jieba.cut(sentence))
        vec = model.sents2elmo([segment])
    if predict:
        print(segment)
    return [np.mean(v, axis=0) for v in vec]


def cal_cosine(a, b):
    """
    计算余弦相似度
    :param a: 张量 a
    :param b: 张量 b
    :return: 余弦相似度
    """
    return np.matmul(a, np.array(b).T) / np.linalg.norm(a) / np.linalg.norm(b, axis=1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="word2vec", type=str, required=False,
                        help="using word2vec or elmo")
    args = parser.parse_args()
    print('using model:', args.model)
    model_type = args.model
    question_vec = []
    model = None
    if model_type == 'elmo':
        # 使用elmo生成句向量
        model = Embedder(os.path.join(os.getcwd(), 'elmo'))
        if os.path.exists('data/elmo_embedding.pkl'):
            question_vec = joblib.load('data/elmo_embedding.pkl')
        else:
            question_vec.extend(elmo2vec(model, question))
            joblib.dump(question_vec, 'data/elmo_embedding.pkl')
    elif model_type == 'word2vec':
        # 使用word2vec生成句向量
        temporary_filepath = 'word2vec/wiki.model'
        model = gensim.models.Word2Vec.load(temporary_filepath)
        for q in question:
            question_vec.append(sen2vec(model, q))

    while 1:
        # 用户的问题转向量
        q = input('问题：')
        vec = None
        if model_type == 'elmo':
            vec = elmo2vec(model, q, predict=True)[0]
        elif model_type == 'word2vec':
            vec = sen2vec(model, q, predict=True)
        print(vec)
        cosine = cal_cosine(vec, question_vec)
        # 最大值
        max_cosine = max(cosine)
        # 最大值对应的索引
        index = np.argmax(cosine)
        print(max_cosine)
        if max_cosine < 0.8:
            print('没有找到准确的答案，你想问的问题是不是：', question[index])
            continue
        print('答案：', answer[index])
