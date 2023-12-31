import os
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


# 构造轨迹图
def build_traj_graph(df):
    G = nx.DiGraph()
    users = list(set(df['user'].to_list()))
    loop = tqdm(users)
    for user_id in loop:
        user_df = df[df['user'] == user_id]
        # 遍历每一个用户的每一条轨迹，将其签到的POI作为节点添加到图中，并记录该POI被签到的次数
        for i, row in user_df.iterrows():
            node = row['poi']
            if node not in G.nodes():   # 加入轨迹图中，初始化签到次数为1
                G.add_node(row['poi'], checkin_cnt=1)
            else:   # 已在图中，签到次数+1
                G.nodes[node]['checkin_cnt'] += 1
        # 对于每一条轨迹，按照签到顺序，添加有向边
        previous_poi_id = 0
        previous_traj_id = 0
        for i, row in user_df.iterrows():
            poi_id = row['poi']
            traj_id = row['seq']
            # 轨迹的第一个节点以及不同的轨迹的结束和开始之间没有边
            if (previous_poi_id == 0) or (previous_traj_id != traj_id):
                previous_poi_id = poi_id
                previous_traj_id = traj_id
                continue
            # 添加有向边
            if G.has_edge(previous_poi_id, poi_id):
                G.edges[previous_poi_id, poi_id]['weight'] += 1
            else:
                G.add_edge(previous_poi_id, poi_id, weight=1)
            previous_traj_id = traj_id
            previous_poi_id = poi_id
    return G


# 构造交通网络图
def build_trans_graph(df):
    G = nx.DiGraph()
    trans = list(set(df['trans_id'].to_list()))
    loop = tqdm(trans)
    for tran_id in loop:
        tran_df = df[df['trans_id'] == tran_id]
        # 遍历每条线路，将站点作为节点添加到图中，并记录该站点经过的次数
        for i, row in tran_df.iterrows():
            node = row['id']
            if node not in G.nodes():   # 加入轨迹图中，初始化次数为1
                G.add_node(row['id'],
                           cnt=1,
                           name=row['name'],
                           latitude=row['latitude'],
                           longitude=row['longitude'])
            else:   # 已在图中，签到次数+1
                G.nodes[node]['cnt'] += 1
        # 对于每一条线路，按照顺序，添加有向边
        previous_poi_id = 0
        previous_trans_id = 0
        for i, row in tran_df.iterrows():
            poi_id = row['id']
            traj_id = row['seq']
            # 轨迹的第一个节点以及不同的轨迹的结束和开始之间没有边
            if (previous_poi_id == 0) or (previous_trans_id != traj_id):
                previous_poi_id = poi_id
                previous_trans_id = traj_id
                continue
            # 添加有向边
            if G.has_edge(previous_poi_id, poi_id):
                G.edges[previous_poi_id, poi_id]['weight'] += 1
            else:
                G.add_edge(previous_poi_id, poi_id, weight=1)
            previous_trans_id = traj_id
            previous_poi_id = poi_id
    return G


# 保存轨迹图到csv
def traj2csv(G):
    dst_dir = '../data/'
    nodelist = G.nodes()
    # 以邻接矩阵形式保存
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    np.savetxt(os.path.join(dst_dir, 'traj_graph_A.csv'), A.todense(), delimiter=',')
    nodes_data = list(G.nodes.data())
    # 保存poi ID和签到次数
    with open(os.path.join(dst_dir, 'traj_graph_X.csv'), 'w') as f:
        print('poi_id,checkin_cnt', file=f)
        for each in nodes_data:
            print(each)
            node_name = each[0]
            checkin_cnt = each[1]['checkin_cnt']
            print(f'{node_name},{checkin_cnt}', file=f)
            

# 保存交通网络图到csv
def trans2csv(G):
    dst_dir = '../data/'
    nodelist = G.nodes()
    # 以邻接矩阵形式保存
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    np.savetxt(os.path.join(dst_dir, 'trans_graph_A.csv'), A.todense(), delimiter=',')
    nodes_data = list(G.nodes.data())
    # 保存poi ID和签到次数
    with open(os.path.join(dst_dir, 'trans_graph_X.csv'), 'w') as f:
        print('poi_id,cnt, name, latitude, longitude', file=f)
        for each in nodes_data:
            print(each)
            node_name = each[0]
            checkin_cnt = each[1]['cnt']
            name = each[1]['name']
            latitude = each[1]['latitude']
            longitude = each[1]['longitude']
            print(f'{node_name},{checkin_cnt}, {name}, {latitude}, {longitude}', file=f)


# 加载POI轨迹邻接矩阵
def load_graph_adj_mtx(path):
    # A是(num, num)的矩阵，内容是weight转移概率
    A = np.loadtxt(path, delimiter=',')
    return A


# 加载POI特征
def load_poi_features(path):
    df = pd.read_csv(path)
    X = df.to_numpy()
    return X


if __name__ == '__main__':
    # traj_df = pd.read_csv(os.path.join('../data/traj.csv'))
    # print('----------Building trajectory graph ----------')
    # G = build_traj_graph(traj_df)
    # traj2csv(G)

    trans_df = pd.read_csv(os.path.join('../data/transportation.csv'), encoding='ANSI')
    print('----------Building transportation graph ----------')
    G = build_trans_graph(trans_df)
    trans2csv(G)
