import torch
import torch.nn as nn
import torch.nn.functional as F

from sim_model.base_model import BaseModel


class ESIM(BaseModel):
    def __init__(self,
                 num_perspective=20,
                 char_vocab_size=100,
                 char_dim=10,
                 char_hidden_size=10,
                 hidden_size=10,
                 max_word_len=10,
                 dropout_rate=0.2,
                 training=True
                 ):
        super(ESIM, self).__init__()

        self.max_word_len = max_word_len
        self.char_hidden_size = char_hidden_size
        self.dropout_rate = dropout_rate
        self.training = training

        # representation
        # self.d = word_dim + char_hidden_size
        self.d = char_hidden_size
        # perspective
        self.l = num_perspective

        # Word Representation Layer
        self.char_embedding = nn.Embedding(char_vocab_size, char_dim)
        # self.word_embedding = nn.Embedding(word_vocab_size, word_dim)

        self.char_LSTM = nn.LSTM(
            input_size=char_dim,
            hidden_size=char_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True)

        # Context Representation Layer
        self.context_LSTM = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True)

        # ----- Prediction Layer -----
        self.fc1 = nn.Linear(2048, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

    def dropout(self, v):
        return F.dropout(v, p=self.dropout_rate, training=self.training)

    def forward(self, char_p, char_q):
        p_embedding, _ = self.char_LSTM(self.char_embedding(char_p.long()))
        q_embedding, _ = self.char_LSTM(self.char_embedding(char_q.long()))

        p_embedding = self.dropout(p_embedding)
        q_embedding = self.dropout(q_embedding)

        # attention
        e = torch.matmul(p_embedding, torch.transpose(q_embedding, 1, 2))
        p_hat = torch.matmul(torch.softmax(e, dim=2), q_embedding)
        q_hat = torch.matmul(torch.softmax(e, dim=1), p_embedding)

        p_cat = torch.cat([p_embedding, p_hat, p_embedding - p_hat, p_embedding * p_hat], dim=2)
        q_cat = torch.cat([q_embedding, q_hat, q_embedding - q_hat, q_embedding * q_hat], dim=2)

        p, _ = self.context_LSTM(p_cat)
        q, _ = self.context_LSTM(q_cat)

        p_max = torch.sum(p, dim=1)
        q_max = torch.sum(q, dim=1)
        p_mean = torch.mean(p, dim=1)
        q_mean = torch.mean(q, dim=1)

        x = torch.cat([p_max, q_max, p_mean, q_mean], dim=1)
        x = self.dropout(x)

        # ----- Prediction Layer -----
        x = F.tanh(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x