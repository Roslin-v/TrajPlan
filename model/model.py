import torch.nn as nn


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


# Trajectory Planning Network
class TPN():
    def __init__(self):
        self.node_attn_net()
        self.heuristics_net()

    # 可观测成本g
    def node_attn_net(self):
        # ========== 根据交通网络图计算路线距离
        # ========== 根据已有轨迹计算路线吸引力
        # ========== 两部分合起来计算可观测成本
        print('Finished node attention network.')

    # 估计成本h
    def heuristics_net(self):
        # ========== 利用图神经网络估计到目的地的代价
        # ========== 引入多头注意力机制
        print('Finished heuristics network.')


# Improved A Star Algorithm using TPN
class AStarPlus():
    def __init__(self):
        print('Finished A Star Algorithm.')
