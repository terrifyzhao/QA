import pandas as pd
import gensim.models
import jieba
import numpy as np

df = pd.read_csv('qa_data.csv')

temporary_filepath = 'wiki.model'
model = gensim.models.Word2Vec.load(temporary_filepath)

question = df['question'].values
answer = df['answer'].values


def sen2vec(model, sentence, predict=False):
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


# 把所有的保险问题转换成句向量
vec = []
for q in question:
    vec.append(sen2vec(model, q))


# 余弦相似度
def cal_cosine(a, b):
    return np.matmul(a, np.array(b).T) / np.linalg.norm(a) / np.linalg.norm(b, axis=1)


while 1:
    # 用户的问题转向量
    q = input('问题：')
    v = sen2vec(model, q, predict=True)

    cosine = cal_cosine(v, vec)
    # 最大值
    max_cosine = max(cosine)
    # 最大值对应的索引
    index = np.argmax(cosine)
    print(max_cosine)
    if max_cosine < 0.9:
        print('没有找到准确的答案，你想问的问题是不是：', question[index])
        continue
    print('答案：', answer[index])
