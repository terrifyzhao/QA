import torch
import torch.nn as nn
import torch.nn.functional as F
from sim_model.base_model import BaseModel


class BIMPM(BaseModel):
    def __init__(self,
                 num_perspective=20,
                 char_vocab_size=100,
                 char_dim=10,
                 word_vocab_size=10,
                 word_dim=10,
                 char_hidden_size=10,
                 hidden_size=10,
                 max_word_len=10,
                 dropout_rate=0.2,
                 training=True
                 ):
        super(BIMPM, self).__init__()

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
            bidirectional=False,
            batch_first=True)

        # Context Representation Layer
        self.context_LSTM = nn.LSTM(
            input_size=self.d,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Matching Layer
        for i in range(1, 9):
            setattr(self, f'm_w{i}',
                    nn.Parameter(torch.rand(self.l, hidden_size)))

        # ----- Aggregation Layer -----
        self.aggregation_LSTM = nn.LSTM(
            input_size=self.l * 8,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # ----- Prediction Layer -----
        self.fc1 = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, 2)

    def dropout(self, v):
        return F.dropout(v, p=self.dropout_rate, training=self.training)

    def div_with_small_value(self, n, d, eps=1e-8):
        # too small values are replaced by 1e-8 to prevent it from exploding.
        d = d * (d > eps).float() + eps * (d <= eps).float()
        return n / d

    def full_matching(self, v1, v2, w):
        seq_len = v1.size(1)
        # (1, 1, hidden_size, l)
        w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)
        # (batch, seq_len, hidden_size, l)
        v1 = w * torch.stack([v1] * self.l, dim=3)
        if len(v2.size()) == 3:
            v2 = w * torch.stack([v2] * self.l, dim=3)
        else:
            v2 = w * torch.stack([torch.stack([v2] * seq_len, dim=1)] * self.l, dim=3)

        m = F.cosine_similarity(v1, v2, dim=2)

        return m

    def pairwise_matching(self, v1, v2, w):
        # (1, l, 1, hidden_size)
        w = w.unsqueeze(0).unsqueeze(2)
        # (batch, l, seq_len, hidden_size)
        v1, v2 = w * torch.stack([v1] * self.l, dim=1), w * torch.stack([v2] * self.l, dim=1)
        # (batch, l, seq_len, hidden_size->1)
        v1_norm = v1.norm(p=2, dim=3, keepdim=True)
        v2_norm = v2.norm(p=2, dim=3, keepdim=True)

        # (batch, l, seq_len1, seq_len2)
        n = torch.matmul(v1, v2.transpose(2, 3))
        d = v1_norm * v2_norm.transpose(2, 3)

        # (batch, seq_len1, seq_len2, l)
        m = self.div_with_small_value(n, d).permute(0, 2, 3, 1)

        return m

    def attentive(self, v1, v2):
        # (batch, seq_len1, 1)
        v1_norm = v1.norm(p=2, dim=2, keepdim=True)
        # (batch, 1, seq_len2)
        v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)

        # (batch, seq_len1, seq_len2)
        a = torch.bmm(v1, v2.permute(0, 2, 1))
        d = v1_norm * v2_norm

        return self.div_with_small_value(a, d)

    def forward(self, char_p, char_q):
        # p = self.word_emb(word_p)
        # q = self.word_emb(word_q)

        # (batch, seq_len, max_word_len) -> (batch * seq_len, max_word_len)
        seq_len_p = char_p.size(1)
        seq_len_q = char_q.size(1)

        char_p = char_p.view(-1, self.max_word_len)
        char_q = char_q.view(-1, self.max_word_len)

        # (batch * seq_len, max_word_len, char_dim)-> (1, batch * seq_len, char_hidden_size)
        _, (char_p, _) = self.char_LSTM(self.char_emb(char_p))
        _, (char_q, _) = self.char_LSTM(self.char_emb(char_q))

        # (batch, seq_len, char_hidden_size)
        char_p = char_p.view(-1, seq_len_p, self.char_hidden_size)
        char_q = char_q.view(-1, seq_len_q, self.char_hidden_size)

        # (batch, seq_len, word_dim + char_hidden_size)
        # p = torch.cat([p, char_p], dim=-1)
        # q = torch.cat([q, char_h], dim=-1)

        p = self.dropout(char_p)
        q = self.dropout(char_q)

        # ----- Context Representation Layer -----
        # (batch, seq_len, hidden_size * 2)
        con_p, _ = self.context_LSTM(p)
        con_q, _ = self.context_LSTM(q)

        con_p = self.dropout(con_p)
        con_q = self.dropout(con_q)

        # (batch, seq_len, hidden_size)
        con_p_fw, con_p_bw = torch.split(con_p, self.hidden_size, dim=-1)
        con_q_fw, con_q_bw = torch.split(con_q, self.hidden_size, dim=-1)

        # 1. Full-Matching

        # (batch, seq_len, hidden_size), (batch, hidden_size)
        # -> (batch, seq_len, l)
        p_full_fw = self.full_matching(con_p_fw, con_q_fw[:, -1, :], self.mp_w1)
        p_full_bw = self.full_matching(con_p_bw, con_q_bw[:, 0, :], self.mp_w2)
        q_full_fw = self.full_matching(con_q_fw, con_p_fw[:, -1, :], self.mp_w1)
        q_full_bw = self.full_matching(con_q_bw, con_p_bw[:, 0, :], self.mp_w2)

        # 2. Maxpooling-Matching
        # (batch, seq_len1, seq_len2, l)
        max_fw = self.pairwise_matching(con_p_fw, con_q_fw, self.mp_w3)
        max_qw = self.pairwise_matching(con_p_bw, con_q_bw, self.mp_w4)
        # (batch, seq_len, l)
        p_max_fw, _ = max_fw.max(dim=2)
        p_max_bw, _ = max_qw.max(dim=2)
        q_max_fw, _ = max_fw.max(dim=1)
        q_max_bw, _ = max_qw.max(dim=1)

        # 3. Attentive-Matching

        # (batch, seq_len1, seq_len2)
        att_fw = self.attentive(con_p_fw, con_q_fw)
        att_bw = self.attentive(con_p_bw, con_q_bw)

        # (batch, seq_len2, hidden_size) -> (batch, 1, seq_len2, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_q_fw = con_q_fw.unsqueeze(1) * att_fw.unsqueeze(3)
        att_q_bw = con_q_bw.unsqueeze(1) * att_bw.unsqueeze(3)
        # (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_p_fw = con_p_fw.unsqueeze(2) * att_fw.unsqueeze(3)
        att_p_bw = con_p_bw.unsqueeze(2) * att_bw.unsqueeze(3)

        # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1) -> (batch, seq_len1, hidden_size)
        att_mean_h_fw = self.div_with_small_value(att_q_fw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))
        att_mean_h_bw = self.div_with_small_value(att_q_bw.sum(dim=2), att_bw.sum(dim=2, keepdim=True))

        # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1) -> (batch, seq_len2, hidden_size)
        att_mean_p_fw = self.div_with_small_value(att_p_fw.sum(dim=1), att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1))
        att_mean_p_bw = self.div_with_small_value(att_p_bw.sum(dim=1), att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1))

        # (batch, seq_len, l)
        p_att_mean_fw = self.full_matching(con_p_fw, att_mean_h_fw, self.mp_w5)
        p_att_mean_bw = self.full_matching(con_p_bw, att_mean_h_bw, self.mp_w6)
        q_att_mean_fw = self.full_matching(con_q_fw, att_mean_p_fw, self.mp_w5)
        q_att_mean_bw = self.full_matching(con_q_bw, att_mean_p_bw, self.mp_w6)

        # 4. Max-Attentive-Matching
        # (batch, seq_len1, hidden_size)
        att_max_h_fw, _ = att_q_fw.max(dim=2)
        att_max_h_bw, _ = att_q_bw.max(dim=2)
        # (batch, seq_len2, hidden_size)
        att_max_p_fw, _ = att_p_fw.max(dim=1)
        att_max_p_bw, _ = att_p_bw.max(dim=1)

        # (batch, seq_len, l)
        p_att_max_fw = self.full_matching(con_p_fw, att_max_h_fw, self.mp_w7)
        p_att_max_bw = self.full_matching(con_p_bw, att_max_h_bw, self.mp_w8)
        q_att_max_fw = self.full_matching(con_q_fw, att_max_p_fw, self.mp_w7)
        q_att_max_bw = self.full_matching(con_q_bw, att_max_p_bw, self.mp_w8)

        # (batch, seq_len, l * 8)
        p = torch.cat(
            [p_full_fw, p_max_fw, p_att_mean_fw, p_att_max_fw,
             p_full_bw, p_max_bw, p_att_mean_bw, p_att_max_bw], dim=2)
        q = torch.cat(
            [q_full_fw, q_max_fw, q_att_mean_fw, q_att_max_fw,
             q_full_bw, q_max_bw, q_att_mean_bw, q_att_max_bw], dim=2)

        p = self.dropout(p)
        q = self.dropout(q)

        # ----- Aggregation Layer -----
        # (batch, seq_len, l * 8) -> (2, batch, hidden_size)
        _, (agg_p_last, _) = self.aggregation_LSTM(p)
        _, (agg_q_last, _) = self.aggregation_LSTM(q)

        # 2 * (2, batch, hidden_size) -> 2 * (batch, hidden_size * 2) -> (batch, hidden_size * 4)
        x = torch.cat(
            [agg_p_last.permute(1, 0, 2).contiguous().view(-1, self.args.hidden_size * 2),
             agg_q_last.permute(1, 0, 2).contiguous().view(-1, self.args.hidden_size * 2)], dim=1)
        x = self.dropout(x)

        # ----- Prediction Layer -----
        x = F.tanh(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
