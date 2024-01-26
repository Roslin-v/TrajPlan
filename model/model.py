import torch
import torch.nn as nn
import math


class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )

    def forward(self, user_idx):
        # 将用户索引转换成对应的嵌入向量
        embed = self.user_embedding(user_idx)
        return embed


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


class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), 0))
        x = self.leaky_relu(x)
        return x


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


class Time2Vec(nn.Module):
    def __init__(self, out_dim):
        super(Time2Vec, self).__init__()
        self.out_features = out_dim
        self.w0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.w = nn.parameter.Parameter(torch.randn(1, out_dim - 1))
        self.b = nn.parameter.Parameter(torch.randn(1, out_dim - 1))
        self.f = torch.sin

    def forward(self, x):
        v1 = self.f(torch.matmul(x, self.w) + self.b)
        v2 = torch.matmul(x, self.w0) + self.b0
        return torch.cat([v1, v2], 1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        '''
        num_poi：POI（Point of Interest）的数量。
        num_cat：类别的数量。
        embed_size：嵌入向量的维度。
        nhead：多头注意力的头数。
        nhid：隐藏层的维度。
        nlayers：Transformer 编码器的层数。
        dropout：Dropout 层的丢弃率，默认为 0.5。
        '''
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)  # 给输入序列添加位置编码
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)  # 堆叠多个 Transformer 编码器层
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)   # 线性层，用于将 Transformer 输出转换为 POI 的预测结果
        self.decoder_time = nn.Linear(embed_size, 1)
        self.decoder_cat = nn.Linear(embed_size, num_cat)
        self.init_weights() # 初始化模型参数

    # 生成一个用于遮蔽输入序列中未来位置的掩码
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # 用负无穷填充了非上三角部分，以便在计算注意力时忽略未来位置的信息
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()  # 将偏置项初始化为零
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)    # 将权重项初始化为均匀分布的随机值

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_cat = self.decoder_cat(x)
        return out_poi, out_time, out_cat
