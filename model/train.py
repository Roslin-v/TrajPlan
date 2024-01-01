from data_process import *


def train(model, args):
    # ========== 加载数据集
    print('----------Loading data----------')
    # ========== 数据预处理
    # Step 1: 生成图
    traj_df = pd.read_csv(os.path.join('../data/traj.csv'))
    print('----------Building trajectory graph ----------')
    G = build_traj_graph(traj_df)
    traj2csv(G)
    trans_df = pd.read_csv(os.path.join('../data/transportation.csv'), encoding='ANSI')
    print('----------Building transportation graph ----------')
    G = build_trans_graph(trans_df)
    trans2csv(G)
    # Step 2: 生成字典
    poi_id_dict, cat_id_dict = initiate_dict()
    # Step 3: 上下文嵌入
    # ========== 利用时间差分法训练网络
    print('----------Training----------')
