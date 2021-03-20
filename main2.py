import numpy as np
import pandas as pd
import gensim
import jieba
import os
from elmoformanylangs import Embedder
from text_classification.predict import classification_predict
from chitchat.interact import chitchat
from text_similarity.predict import predict

df = pd.read_csv('data/qa_data.csv')
question = df['question'].values
answer = df['answer'].values

model_path = 'word2vec/wiki.model'
model = gensim.models.Word2Vec.load(model_path)


# model = Embedder(os.path.join(os.getcwd(), 'elmo'))


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


question_vec = []
for q in question:
    question_vec.append(sen2vec(q))


def qa(text):
    # ---------------分类-------------------
    prob = classification_predict(text)
    print('是闲聊的概率是：', prob[0])
    if prob[0] > 0.5:
        print('当前是闲聊')
        res = chitchat(text)
        print(res)
        return res

    # ---------------文本表示-------------------
    vec = sen2vec(text)

    # 计算相似度
    similarity = cosine(vec, question_vec)
    # print('最大的相似度:', max(similarity))
    max_similarity = max(similarity)
    # 最大的相似度对应的下标
    index = np.argmax(similarity)
    if max_similarity < 0.8:
        print(max_similarity)
        print('没有找到对应的问题，你问的是不是:', question[index])
        return f'没有找到对应的问题，你问的是不是:, {question[index]}'

    # ---------------排序-------------------
    top_10 = np.argsort(-similarity)[0:10]
    candidate = question[top_10]
    esim_res = predict([text] * 10, candidate)
    index_dic = {}
    print('候选集：')
    for i, index in enumerate(top_10):
        print(candidate[i], '\t', similarity[index], '\t', esim_res[i])
        index_dic[i] = index

    esim_index = np.argsort(-esim_res)[0]
    print('最相似的问题：', question[index_dic[esim_index]])
    print('答案:', answer[index_dic[esim_index]])
    return answer[index_dic[esim_index]]


if __name__ == '__main__':

    while 1:
        text = input('请输入您的问题：')
        qa(text)
