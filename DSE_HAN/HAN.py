import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

class nodeAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.01, concat=False, is_hetero=False):
        super(nodeAttentionLayer, self).__init__()
        self.is_hetero = is_hetero
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        if self.is_hetero:
            self.Wu = nn.Parameter(torch.empty(size=(in_features, out_features)))
            nn.init.xavier_uniform_(self.Wu.data, gain=1.414)
            self.Wv = nn.Parameter(torch.empty(size=(in_features, out_features)))
            nn.init.xavier_uniform_(self.Wv.data, gain=1.414)
        else:
            self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
            nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, adj, *h):
        if self.is_hetero:
            Whu = torch.mm(h[0], self.Wu)
            Whv = torch.mm(h[1], self.Wv)
            e = self._prepare_attentional_mechanism_input(Whu, Whv)
        else:
            Whv = torch.mm(h[0], self.W)
            e = self._prepare_attentional_mechanism_input(Whv)
                # todo 对于有权图，用e点乘adj

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Whv)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, *wh):
        Wh1 = torch.matmul(wh[0], self.a[:self.out_features, :])
        Wh2 = torch.matmul(wh[-1], self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# if __name__ == '__main__':
    # t1=torch.tensor([[5.5, 6.6, 7.7, 8.8],
    #                   [3, 7.0, 5, 9],
    #                   [6, 8.0, 4, 2]])
    # t2=torch.tensor([[7.7, 6, 5.5, 8.8],
    #                  [2, 8.0, 5, 2],
    #                   [7, 6.0, 5, 4],
    #                  [3, 7.2, 6, 2.3]])
    # adj=torch.tensor([[1, 1, 0, 1],
    #                   [1, 1, 0, 0],
    #                   [0, 0, 1, 1]], dtype=torch.float)
    # att = nodeAttentionLayer(4, 5, dropout=0.1, alpha=0.01, concat=True)
    # res=att(adj, t1, t2)

    # t1=torch.tensor([[5.5, 6.6, 7.7, 8.8],
    #                   [3, 7.0, 5, 9],
    #                   [6, 8.0, 4, 2]])
    # adj=torch.tensor([[1, 0.7, 0],
    #                   [0.5, 1, 0],
    #                   [0, 0, 0.4]], dtype=torch.float)
    # att = nodeAttentionLayer(4, 5, dropout=0.1, alpha=0.01, concat=True, is_hetero=False)
    # res=att(adj, t1)

class nodeAttention(nn.Module):
    def __init__(self, ifeat, ofeat, dropout=0.1, alpha=0.01, nheads=2, is_hetero=False):
        super(nodeAttention, self).__init__()
        self.dropout = dropout
        self.is_hetero = is_hetero

        self.attentions = [nodeAttentionLayer(ifeat, ofeat, dropout=dropout, alpha=alpha, concat=False, is_hetero=is_hetero) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(nhid * nheads, ofeat, dropout=dropout, alpha=alpha, concat=False)
        self.out_fc = nn.Linear(ofeat*nheads, ofeat).float()

    def forward(self, adj, *h):
        x = [F.dropout(x, self.dropout, training=self.training) for x in h]
        # if self.is_hetero:
        #     x = torch.stack([att(adj, x[0], x[1]) for att in self.attentions], dim=1)
        # else:
        #     x = torch.stack([att(adj, x[0]) for att in self.attentions], dim=1)
        if self.is_hetero:
            x = torch.cat([att(adj, x[0], x[1]) for att in self.attentions], dim=1)
        else:
            x = torch.cat([att(adj, x[0]) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_fc(x))
        return x

# if __name__ == '__main__':
#     t1=torch.tensor([[5.5, 6.6, 7.7, 8.8],
#                       [3, 7.0, 5, 9],
#                       [6, 8.0, 4, 2]])
#     adj=torch.tensor([[1, 0.7, 0],
#                       [0.5, 1, 0],
#                       [0, 0, 0.4]], dtype=torch.float)
#     att = nodeAttention(4, 5, nheads=2, is_hetero=False)
#     res=att(adj, t1)
#     print(res)


class SemanticAttentionLayer(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, dropout=0.1):
        super(SemanticAttentionLayer, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )
        self.out_fc = nn.Linear(in_size, out_size)
        self.dropout = dropout

    def forward(self, h):
        z = torch.stack(h, dim=1)  # (N, M, D * K)
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        beta = F.dropout(beta, self.dropout, training=self.training)

        return self.out_fc((beta * z).sum(1))  # (N, D * K)


# if __name__ == '__main__':
#     t1=torch.tensor([[5.5, 6.6, 7.7, 8.8],
#                       [3, 7.0, 5, 9],
#                       [6, 8.0, 4, 2]])
#     t2=torch.tensor([[7.7, 6, 5.5, 8.8],
#                      [2, 8.0, 5, 2],
#                       [7, 6.0, 5, 4],
#                      [3, 7.2, 6, 2.3]])
#     tadj=torch.tensor([[1, 1, 0, 1],
#                       [1, 1, 0, 0],
#                       [0, 0, 1, 1]], dtype=torch.float)
#     tatt = NodeAttention(4, 5)
#
#     h=torch.tensor([[5.5, 6.6, 7.7, 8.8],
#                       [3, 7.0, 5, 9],
#                       [6, 8.0, 4, 2]])
#     hadj2=torch.tensor([[1, 0.7, 0],
#                       [0.5, 1, 0],
#                       [0, 0, 0.4]], dtype=torch.float)
#     hatt = NodeAttention(4, 5, is_hetero=False)
#
#
#     Satt = SemanticAttentionLayer(15,18,5)
#     res = Satt([tatt(tadj, t1, t2), hatt(hadj2, h)])
#     print(res)


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, nheads=2, dropout=0.1):
        super(SemanticAttention, self).__init__()
        self.attentions = [SemanticAttentionLayer(in_size, hidden_size, out_size) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_fc = nn.Linear(out_size*nheads, out_size).float()
        self.dropout = dropout

    def forward(self, h):
        x = torch.cat([att(h) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_fc(x))
        return x


