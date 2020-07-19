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
        try:
            vec += model.wv[s]
        except:
            pass
    vec = vec / len(segment)
    return vec


vec = []
for q in question:
    vec.append(sen2vec(model, q))

while 1:
    q = input('问题：')
    v = sen2vec(model, q, predict=True)

    cosine = np.matmul(v, np.array(vec).T) / np.linalg.norm(v) / np.linalg.norm(vec, axis=1)
    max_cosine = max(cosine)
    index = np.argmax(cosine)
    print(max_cosine)
    if max_cosine < 0.9:
        print('没有找到准确的答案，你想问的问题是不是：', question[index])
        continue
    print('答案：', answer[index])
