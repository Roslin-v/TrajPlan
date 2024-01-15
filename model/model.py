import torch
import torch.nn as nn
import math
from geopy.distance import geodesic


class CategoryEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()
        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
        )

    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        '''
        in_features: 输入特征的维度
        out_features: 输出特征的维度
        bias: 是否使用偏置
        '''
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 可训练参数矩阵，用于将输入特征映射到输出特征
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            # 可训练参数向量，用于增加模型的灵活性
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 初始化模型参数，将权重矩阵和偏置向量（如果有）都初始化为均匀分布的随机值
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        '''
        input: 输入特征向量
        adj: 邻接矩阵，表示节点之间的连接关系
        '''
        # 表示每个节点从其邻居节点收到的信息
        support = torch.mm(input, self.weight)
        # 表示每个节点聚合邻居节点信息后得到的新特征
        output = torch.spmm(adj, support)
        # 如果有偏置，将其加到输出特征上（注意此处的加法是广播加法，会自动将偏置向量复制多份以匹配输出特征的形状）
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()
        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2) # 增强模型的非线性能力

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = nn.functional.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x


class NodeAttnMap(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap, self).__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        # 可学习参数矩阵，用于将输入特征映射到隐藏层维度
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # 可学习参数矩阵，用于计算注意力权重
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        # 带有负数斜率的 LeakyReLU 激活函数
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        '''
        X: 输入特征向量
        A: 邻接矩阵，表示节点之间的连接关系
        '''
        # 经过线性变换后的特征
        Wh = torch.mm(X, self.W)
        # 注意力权重
        e = self._prepare_attentional_mechanism_input(Wh)
        # 将结果进行掩码操作（不相关的位置置零）
        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))  # mask
        A = A + 1  # shift from 0-1 to 1-2
        e = e * A
        return e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


# 多头注意力机制
class MultiHeadedAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        assert args.hidden_size % args.num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = args.hidden_size // args.num_heads
        self.num_heads = args.num_heads
        self.device = torch.device('cpu')
        # self.add_cls = args.add_cls
        self.scale = self.d_k ** -0.5  # 1/sqrt(dk)
        self.temporal_bias_dim = 64
        self.linear_layers = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in range(3)])
        self.dropout = nn.Dropout(p=0)
        self.proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_drop = nn.Dropout(0)
        if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
            self.temporal_mat_bias_1 = nn.Linear(1, self.temporal_bias_dim, bias=True)
            self.temporal_mat_bias_2 = nn.Linear(self.temporal_bias_dim, 1, bias=True)
        elif self.temporal_bias_dim == -1:
            self.temporal_mat_bias = nn.Parameter(torch.Tensor(1, 1))
            nn.init.xavier_uniform_(self.temporal_mat_bias)

    def forward(self, x, padding_masks, future_mask=True, output_attentions=False, batch_temporal_mat=None):
        batch_size, seq_len, d_model = x.shape
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # l(x) --> (B, T, d_model)
        # l(x).view() --> (B, T, head, d_k)
        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (x, x, x))]
        # q, k, v --> (B, head, T, d_k)
        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # (B, head, T, T)
        batch_temporal_mat = 1.0 / torch.log(torch.exp(torch.tensor(1.0).to(self.device)) + batch_temporal_mat)
        if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
            batch_temporal_mat = self.temporal_mat_bias_2(nn.functional.leaky_relu(
                self.temporal_mat_bias_1(batch_temporal_mat.unsqueeze(-1)),
                negative_slope=0.2)).squeeze(-1)  # (B, T, T)
        if self.temporal_bias_dim == -1:
            batch_temporal_mat = batch_temporal_mat * self.temporal_mat_bias.expand((1, seq_len, seq_len))
        batch_temporal_mat = batch_temporal_mat.unsqueeze(1)  # (B, 1, T, T)
        scores += batch_temporal_mat  # (B, 1, T, T)
        if padding_masks is not None:
            scores.masked_fill_(padding_masks == 0, float('-inf'))
        if future_mask:
            mask_postion = torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).bool().to(self.device)
            if self.add_cls:
                mask_postion[:, 0, :] = 0
            scores.masked_fill_(mask_postion, float('-inf'))
        p_attn = nn.functional.softmax(scores, dim=-1)  # (B, head, T, T)
        p_attn = self.dropout(p_attn)
        out = torch.matmul(p_attn, value)  # (B, head, T, d_k)
        # 3) "Concat" using a view and apply a final linear.
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)  # (B, T, d_model)
        out = self.proj(out)  # (B, T, N, D)
        out = self.proj_drop(out)
        if output_attentions:
            return out, p_attn  # (B, T, dim_out), (B, head, T, T)
        else:
            return out, None  # (B, T, dim_out)


# Trajectory Planning Network
class TPN(nn.Module):
    def __init__(self, args):
        super(TPN, self).__init__()
        self.args = args
        # ========== 可观测成本g

        # Step2: 使用GCN将POI嵌入到保留POI之间全局转换的潜在空间中
        self.poi_embed_model = GCN(ninput=self.args.gcn_nfeat, nhid=self.args.gcn_nhid, noutput=self.args.poi_embed_dim,
                              dropout=self.args.gcn_dropout)
        # Step3: 用转移注意图来模拟从一个POI到另一个POI的转移概率
        self.node_attn_model = NodeAttnMap(in_features=self.args.gcn_nfeat, nhid=self.args.node_attn_nhid, use_mask=False)

        self.rnn = nn.RNN(self.args.poi_num * 2, self.args.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.args.hidden_size, self.args.poi_num)
        # ========== 估计成本h
        # Step1: 利用图神经网络估计到目的地的代价
        # Step2: 引入多头注意力机制

    def forward(self, X, A):
        '''
        combined_input = torch.cat((distance, probability), dim=1)
        g, hidden = self.rnn(combined_input)
        g = self.fc(g)
        '''
        g = self.node_attn_model(X, A)
        g = nn.functional.softmax(g)
        return g
