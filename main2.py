import numpy as np
import pandas as pd
import gensim
import jieba
import os
from elmoformanylangs import Embedder

df = pd.read_csv('data/qa_data.csv')
question = df['question'].values
answer = df['answer'].values

# model_path = 'word2vec/wiki.model'
# model = gensim.models.Word2Vec.load(model_path)

model = Embedder(os.path.join(os.getcwd(), 'elmo'))


def sen2vec(sentence):
    # 我爱NLP
    # ['我','爱','NLP']
    segment = list(jieba.cut(sentence))
    vec = np.zeros(100)
    for s in segment:
        try:
            vec += model.wv[s]
        except:
            # oov 不在词典内的词
            pass
    vec = vec / len(segment)
    return vec


def elmo2vec(sentence):
    if isinstance(sentence, str):
        segment = list(jieba.cut(sentence))
        vec = model.sents2elmo([segment])
    elif isinstance(sentence, np.ndarray):
        segment = [jieba.cut(s) for s in sentence]
        vec = model.sents2elmo(segment)
    return [np.mean(v, axis=0) for v in vec]


def cosine(a, b):
    return np.matmul(a, np.array(b).T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))


if __name__ == '__main__':

    # question_vec = []
    # for q in question:
    #     question_vec.append(sen2vec(q))

    question_vec = elmo2vec(question)

    while 1:
        text = input('请输入您的问题：')

        vec = elmo2vec(text)

        # 计算相似度
        similarity = cosine(vec, question_vec)[0]
        # print('最大的相似度:', max(similarity))
        max_similarity = max(similarity)
        # 最大的相似度对应的下标
        index = np.argmax(similarity)
        if max_similarity < 0.8:
            print(max_similarity)
            print('没有找到对应的问题，你问的是不是:', question[index])
            continue
        print('答案：', answer[index])
