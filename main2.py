import pandas as pd
import numpy as np
import gensim
import jieba
from text_classification.predict import classification_predict
from text_similarity.predict import predict
from chitchat.interact import chitchat

df = pd.read_csv('data/qa_data.csv')
question = df['question'].values
answer = df['answer'].values

model = gensim.models.Word2Vec.load('word2vec/wiki.model')


def sen2vec(text):
    segment = list(jieba.cut(text))
    vec = np.zeros(100)
    for s in segment:
        # 假如我们的词不在词向量里，就会出现oov的问题
        try:
            vec += model.wv[s]
        except:
            pass
    vec = vec / len(segment)
    return vec


def cosine(a, b):
    return np.matmul(a, b.T) / np.linalg.norm(a) / np.linalg.norm(b, axis=-1)


question_vec = []
for q in question:
    question_vec.append(sen2vec(q))


def qa(text):
    # 先判断是否是闲聊
    prob = classification_predict(text)
    if prob > 0.5:
        print('闲聊')
        res = chitchat(text)
        print(res)
        return res
    vec = sen2vec(text)

    # 召回
    similarity = cosine(vec, np.array(question_vec))
    max_similarity = max(similarity)
    print('最大相似度', max_similarity)
    if max_similarity < 0.8:
        print('没有找到答案')
        return '没有找到答案'
    top_10 = np.argsort(-similarity)[0:10]
    candidate = question[top_10]

    # 精排
    esim_res = predict([q] * 10, candidate)

    index_dic = {}

    print('候选集：')
    for i, index in enumerate(top_10):
        print(candidate[i], ' ', similarity[index], ' ', esim_res[i])
        index_dic[i] = index

    esim_index = np.argsort(-esim_res)[0]
    print('最相似的问题: ', question[index_dic[esim_index]])
    print('答案: ', answer[index_dic[esim_index]])
    return answer[index_dic[esim_index]]


if __name__ == '__main__':
    while 1:
        q = input('请输入你的问题：')
        qa(q)
