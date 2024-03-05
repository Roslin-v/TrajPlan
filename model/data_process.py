import os
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse.linalg import eigsh
from geopy.distance import geodesic
import csv


# 构造轨迹图
def build_traj_graph():
    G = nx.DiGraph()
    user_visit = {}  # 每个用户访问过哪些景点
    spot_visit = {}  # 每个景点被哪些用户访问
    # 把所有的POI都加到有向图中
    spot_df = pd.read_csv(os.path.join('../data/spot.csv'))
    for row in spot_df.iterrows():
        G.add_node(row[1][0], checkin_cnt=1)
        spot_visit[row[1][0]] = []
    # 添加全连接的边
    '''
    all_pois = list(set(spot_df['id'].to_list()))
    for i in range(len(all_pois)):
        for j in range(i + 1, len(all_pois)):
            G.add_edge(all_pois[i], all_pois[j], weight=1)
            G.add_edge(all_pois[j], all_pois[i], weight=1)
    '''
    df = pd.read_csv(os.path.join('../data/traj.csv'))
    users = list(set(df['user'].to_list()))
    loop = tqdm(users)
    for user_id in loop:
        user_df = df[df['user'] == user_id]
        user_visit[user_id] = []
        # 对于每一条轨迹，按照签到顺序，添加有向边
        previous_poi_id = 0
        previous_traj_id = 0
        for i, row in user_df.iterrows():
            poi_id = row['poi']
            traj_id = row['seq']
            # 轨迹的第一个节点以及不同的轨迹的结束和开始之间没有边
            if (previous_poi_id == 0) or ((previous_traj_id+1) != traj_id):
                user_visit[user_id].append(poi_id)
                spot_visit[poi_id].append(user_id)
                previous_poi_id = poi_id
                previous_traj_id = traj_id
                continue
            if previous_poi_id == poi_id:
                continue
            # 添加有向边
            if G.has_edge(previous_poi_id, poi_id):
                G.edges[previous_poi_id, poi_id]['weight'] += 1
            else:
                G.add_edge(previous_poi_id, poi_id, weight=1)
            user_visit[user_id].append(poi_id)
            spot_visit[poi_id].append(user_id)
            previous_traj_id = traj_id
            previous_poi_id = poi_id
    return G, user_visit, spot_visit


# 构造交通网络图
def build_trans_graph():
    G = nx.DiGraph()
    df = pd.read_csv(os.path.join('../data/transportation.csv'))
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
            trans_id = row['seq']
            # 轨迹的第一个节点以及不同的轨迹的结束和开始之间没有边
            if (previous_poi_id == 0) or ((previous_trans_id+1) != trans_id):
                previous_poi_id = poi_id
                previous_trans_id = trans_id
                continue
            # 添加有向边，权重是距离
            if not G.has_edge(previous_poi_id, poi_id):
                w = geodesic((float(G.nodes[previous_poi_id]['latitude']), float(G.nodes[previous_poi_id]['longitude'])), (float(G.nodes[poi_id]['latitude']), float(G.nodes[poi_id]['longitude']))).m
                G.add_edge(previous_poi_id, poi_id, weight=w)
            previous_trans_id = trans_id
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


def generate_time():
    spot_df = pd.read_csv(os.path.join('../data/spot.csv'))
    traj_df = pd.read_csv(os.path.join('../data/traj.csv'))
    users = list(set(traj_df['user'].to_list()))
    index = 0
    for user_id in users:
        user_df = traj_df[traj_df['user'] == user_id]
        day_count = 1
        visit = []
        for i, row in user_df.iterrows():
            day_id = row['day']
            if day_id != day_count or i == (user_df.shape[0] + index - 1):
                if i == (user_df.shape[0] + index - 1):
                    visit.append(row['poi'])
                    index += user_df.shape[0]
                recommend_time = []
                for each in visit:
                    recommend_time.append(spot_df[spot_df['id'] == each].iloc[0]['recommend_time'])
                sum_time = sum(recommend_time)
                for j in range(len(recommend_time)):
                    recommend_time[j] /= sum_time
                    recommend_time[j] *= 12
                visit_time = [9]
                csvFile = open('../data/test3.csv', "a", newline='')
                writer = csv.writer(csvFile)
                writer.writerow([0, visit_time[0]])
                for j in range(1, len(recommend_time)):
                    visit_time.append(visit_time[j-1]+recommend_time[j-1])
                    writer.writerow([j, visit_time[j]])
                if i != (user_df.shape[0] + index - 1):
                    day_count = day_id
                    visit = [row['poi']]
            else:
                visit.append(row['poi'])


# 加载转移矩阵
def load_matrix(path):
    # A是(num, num)的矩阵，内容是路线吸引力
    A = np.loadtxt(path, delimiter=',')
    return A


# 加载POI特征
def load_poi_features(path):
    df = pd.read_csv(path)
    X = df.to_numpy()
    return X


# 生成字典
def initiate_dict():
    spot_df = pd.read_csv('./data/spot.csv')
    # food_df = pd.read_csv('../data/food.csv', encoding='ANSI')
    # trans_df = pd.read_csv('../data/transportation.csv', encoding='ANSI')
    # 生成POI ID字典
    spot_ids = list(set(spot_df['id'].tolist()))
    # food_ids = list(set(food_df['id'].tolist()))
    # trans_ids = list(set(trans_df['id'].tolist()))
    poi_ids = spot_ids
    poi_id_dict = dict(zip(poi_ids, range(len(poi_ids))))
    # 生成POI种类字典
    spot_ids = list(set(spot_df['category'].tolist()))
    # food_ids = list(set(food_df['category_id'].tolist()))
    # trans_ids = list(set(trans_df['category'].tolist()))
    cat_ids = spot_ids
    cat_id_dict = dict(zip(cat_ids, range(len(cat_ids))))
    # 生成POI ID：CAT ID字典
    poi_cat_dict = {}
    for i, row in spot_df.iterrows():
        poi_cat_dict[poi_id_dict[row['id']]] = cat_id_dict[row['category']]
    # 生成用户ID字典
    traj_df = pd.read_csv('./data/traj.csv')
    user_ids = [str(each) for each in list(set(traj_df['user'].to_list()))]
    user_id_dict = dict(zip(user_ids, range(len(user_ids))))
    return poi_id_dict, cat_id_dict, poi_cat_dict, user_id_dict


# 计算拉普拉斯矩阵
def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    deg_mat = deg_mat_row
    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))
    if mat_type == 'com_lap_mat':
        # Combinatorial
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat
    elif mat_type == 'wid_rw_normd_lap_mat':
        # For ChebConv
        rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)
        rw_normd_lap_mat = id_mat - rw_lap_mat
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat
        return wid_rw_normd_lap_mat
    elif mat_type == 'hat_rw_normd_lap_mat':
        # For GCNConv
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')


def maksed_mse_loss(input, target, mask_value=-1):
    mask = target == mask_value
    out = (input[~mask] - target[~mask]) ** 2
    loss = out.mean()
    return loss


def top_k_acc_last_timestep(y_true_seq, y_pred_seq, k):
    """ next poi metrics """
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    top_k_rec = y_pred.argsort()[-k:][::-1]
    idx = np.where(top_k_rec == y_true)[0]
    if len(idx) != 0:
        return 1
    else:
        return 0


def mAP_metric_last_timestep(y_true_seq, y_pred_seq, k):
    """ next poi metrics """
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-k:][::-1]
    r_idx = np.where(rec_list == y_true)[0]
    if len(r_idx) != 0:
        return 1 / (r_idx[0] + 1)
    else:
        return 0


def MRR_metric_last_timestep(y_true_seq, y_pred_seq):
    """ next poi metrics """
    # Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-len(y_pred):][::-1]
    r_idx = np.where(rec_list == y_true)[0][0]
    return 1 / (r_idx + 1)


if __name__ == '__main__':
    '''
    print('----------Building trajectory graph ----------')
    G, _, _ = build_traj_graph()
    traj2csv(G)
    trans_df = pd.read_csv(os.path.join('../data/transportation.csv'), encoding='ANSI')
    print('----------Building transportation graph ----------')
    G = build_trans_graph(trans_df)
    trans2csv(G)
    initiate_dict()
    '''
    generate_time()
