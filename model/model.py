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


# Trajectory Planning Network
class TPN(nn.Module):
    def __init__(self, args):
        super(TPN, self).__init__()
        self.args = args
        # ========== 可观测成本g
        # Step1: 根据交通网络图计算路线距离
        self.dis_list = []
        # Step2: 使用GCN将POI嵌入到保留POI之间全局转换的潜在空间中
        self.poi_embed_model = GCN(ninput=self.args.gcn_nfeat, nhid=self.args.gcn_nhid, noutput=self.args.poi_embed_dim,
                              dropout=self.args.gcn_dropout)
        # Step3: 用转移注意图来模拟从一个POI到另一个POI的转移概率
        self.node_attn_model = NodeAttnMap(in_features=self.args.gcn_nfeat, nhid=self.args.node_attn_nhid, use_mask=False)
        # Step4: 距离+转移概率计算可观测成本
        self.fc1 = nn.Linear()
        self.relu1 = nn.ReLU()
        self.loss1 = nn.Softmax()
        # ========== 估计成本h
        # Step1: 利用图神经网络估计到目的地的代价
        # Step2: 引入多头注意力机制

    def forward(self, X, A):
        # Todo: 输入我是先随便写的。。记得改一下
        # poi_embed = self.poi_embed_model(X)
        # 计算两两之间的距离
        for i in range(len(X)-1):
            self.dis_list.append(geodesic(X[i], X[i+1]).m)
        distance = torch.Tensor(self.dis_list).detach()  # detach函数抽离出来，反向传播不会更新它
        node_attn = self.node_attn_model(X, A)
        g = self.fc1(torch.cat((distance, node_attn), dim=2))
        g = self.relu1(g)
        g = self.loss1(g)
        return g


# Improved A Star Algorithm using TPN
class AStarPlus():
    def __init__(self):
        print('Finished A Star Algorithm.')
