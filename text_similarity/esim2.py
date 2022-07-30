import torch
import torch.nn as nn


class ESIM(nn.Module):
    def __init__(self,
                 vocab_size,
                 char_dim,
                 char_hidden_size,
                 max_len):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, char_dim)

        # [batch_size, seq_len, hidden_size]
        # [seq_len, batch_size, hidden_size]
        self.char_lstm = nn.LSTM(input_size=char_dim,
                                 hidden_size=char_hidden_size,
                                 num_layers=1,
                                 bidirectional=True,
                                 batch_first=True)

        self.context_lstm = nn.LSTM(input_size=char_hidden_size * 8,
                                    hidden_size=char_hidden_size,
                                    num_layers=1,
                                    bidirectional=True,
                                    batch_first=True)

        self.max_pool = nn.MaxPool2d(kernel_size=(max_len, 1))

        self.fc1 = nn.Linear(char_hidden_size * 8, char_hidden_size)
        self.fc2 = nn.Linear(char_hidden_size, 1)

        self.drop_out = nn.Dropout(0.2)

    def forward(self, char_p, char_q):
        # Input Enconding
        embedding_p = self.embedding(char_p)
        embedding_q = self.embedding(char_q)
        lstm_p, _ = self.char_lstm(embedding_p)
        lstm_q, _ = self.char_lstm(embedding_q)
        lstm_p = self.drop_out(lstm_p)
        lstm_q = self.drop_out(lstm_q)

        # Local Inference Modeling
        # [batch_size, seq_len, hidden_size] * [batch_size, hidden_size, seq_len]
        # [batch_size, seq_len_p, seq_len_q]
        e = torch.matmul(lstm_p, torch.transpose(lstm_q, 1, 2))
        p_hat = torch.matmul(torch.softmax(e, dim=2), lstm_q)
        q_hat = torch.matmul(torch.transpose(torch.softmax(e, dim=1), 1, 2), lstm_p)

        p_cat = torch.cat([lstm_p, p_hat, lstm_p - p_hat, lstm_p * p_hat], dim=-1)
        q_cat = torch.cat([lstm_q, q_hat, lstm_q - q_hat, lstm_q * q_hat], dim=-1)

        # Inference Composition
        p = self.context_lstm(p_cat)[0]
        q = self.context_lstm(q_cat)[0]

        # Predict
        p_max = self.max_pool(p).squeeze(dim=1)
        q_max = self.max_pool(q).squeeze(dim=1)

        p_mean = torch.mean(p, dim=1)
        q_mean = torch.mean(q, dim=1)

        y = torch.cat([p_max, q_max, p_mean, q_mean], dim=-1)
        y = self.drop_out(y)

        y = self.fc1(y)
        y = torch.tanh(y)
        y = self.drop_out(y)
        y = self.fc2(y)
        y = torch.sigmoid(y)
        return y.squeeze(dim=-1)
