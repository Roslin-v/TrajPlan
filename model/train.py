import os.path
import numpy as np
import torch
from data_process import *
from model import *
from sklearn.preprocessing import OneHotEncoder
import pickle
import torch.optim as optim


def train(args):
    # ========== 加载数据集
    print('----------Loading data----------')
    # ========== 数据预处理
    # Step 1: 生成图
    # traj_df = pd.read_csv(os.path.join('../data/traj.csv'))
    # print('----------Building trajectory graph ----------')
    # G = build_traj_graph(traj_df)
    # traj2csv(G)
    # trans_df = pd.read_csv(os.path.join('../data/transportation.csv'), encoding='ANSI')
    # print('----------Building transportation graph ----------')
    # G = build_trans_graph(trans_df)
    # trans2csv(G)
    # Step 2: 生成字典
    poi_id_dict, cat_id_dict = initiate_dict()
    # Step 3: 加载数据
    # 邻接矩阵，内容是一个POI到另一个POI的转移次数
    raw_A = load_graph_adj_mtx(args.data_adj_mtx)
    # poi_id, name, cat, score, comment, price, lat, long, recommend_time, night_visit, checkin_cnt
    raw_X = load_poi_features(args.data_node_feats)
    num_pois = raw_X.shape[0]
    # Step 4: one hot编码
    one_hot_encoder = OneHotEncoder()
    cat_list = list(raw_X[:, 2])
    one_hot_encoder.fit(list(map(lambda x: [x], cat_list)))
    one_hot_rlt = one_hot_encoder.transform(list(map(lambda x: [x], cat_list))).toarray()
    num_cats = one_hot_rlt.shape[-1]
    X = np.zeros((num_pois, raw_X.shape[-1] - 1 + num_cats), dtype=np.float32)
    X[:, 0] = raw_X[:, 0]
    X[:, 2:num_cats + 2] = one_hot_rlt  # 把one hot编码插入到原始数据的第三列
    X[:, num_cats + 2:] = raw_X[:, 3:]
    with open(os.path.join('../data/one-hot-encoder.pkl'), 'wb') as f:
        pickle.dump(one_hot_encoder, f)
    # 正则化
    # A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')
    A = raw_A
    # 计算两两之间的距离
    '''
    coord = X[:, 7:9]
    distance = torch.zeros(args.poi_num, args.poi_num)
    for i in range(args.poi_num):
        for j in range(i + 1, args.poi_num):
            d = geodesic((coord[i][0], coord[i][1]), (coord[j][0], coord[j][1])).m
            distance[i][j] = distance[j][i] = d
    np.savetxt(os.path.join('../data/distance.csv'), distance.detach().numpy(), delimiter=',')
    '''
    distance = load_distance('../data/distance.csv')
    # Step 5: 上下文嵌入
    # cat_embed = CategoryEmbeddings(num_cats, args.cat_embed_dim)
    # ========== 初始化模型
    '''
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        A = torch.from_numpy(A)
        distance = torch.from_numpy(distance)
    X = X.to(device=torch.device('cpu'), dtype=torch.float)
    A = A.to(device=torch.device('cpu'), dtype=torch.float)
    distance = distance.to(device=torch.device('cpu'), dtype=torch.float)
    args.poi_num = X.shape[0]       # POI个数
    args.gcn_nfeat = X.shape[1]     # POI特征个数
    tpn = TPN(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(tpn.parameters(), lr=args.lr)
    optimizer.zero_grad()
    # ========== 训练网络
    print('----------Training----------')
    temp = raw_A - np.ones((args.poi_num, args.poi_num))
    np.fill_diagonal(temp, 0)
    target = calculate_laplacian_matrix(temp, mat_type='hat_rw_normd_lap_mat')
    # target = temp
    target = torch.from_numpy(target)
    target = target.to(device=torch.device('cpu'), dtype=torch.float)
    outputs = torch.zeros(args.poi_num, args.poi_num)
    for epoch in range(200):
        outputs = tpn(X, A)
        loss = criterion(outputs.view(args.poi_num*args.poi_num), target.view(args.poi_num*args.poi_num))
        loss.backward()
        optimizer.step()
        print(loss)
    print(outputs)
    '''
