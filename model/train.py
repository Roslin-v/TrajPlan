import os.path
import logging
from .data_process import *
from .model import *
from sklearn.preprocessing import OneHotEncoder
import pickle
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import re
import glob


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def train(args):
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, sep='-')
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(args.save_dir, f"log_training.txt"),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.info(args)

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
    # Step 2: 加载POI数据
    # poi_id, name, cat, score, comment, price, lat, long, recommend_time, night_visit, checkin_cnt
    raw_X = load_poi_features(args.data_node_feats)
    num_pois = raw_X.shape[0]
    # Step 3: one hot编码
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
    raw_A = load_matrix(args.data_adj_mtx)
    A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')
    # Step 4: 生成字典
    poi_id_dict, cat_id_dict, poi_cat_dict, user_id_dict = initiate_dict()

    # ========== 加载数据集
    class TrajectoryDatasetTrain(Dataset):
        def __init__(self, train_df):
            self.df = train_df
            self.traj_seqs = []
            self.input_seqs = []
            self.label_seqs = []

            for traj_id in tqdm(set(train_df['user'].tolist())):
                traj_df = train_df[train_df['user'] == traj_id]
                poi_ids = traj_df['poi'].to_list()
                poi_idxs = [poi_id_dict[each] for each in poi_ids]
                time_feature = traj_df['norm'].to_list()
                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))
                if len(input_seq) < args.short_traj_thres:
                    continue
                self.traj_seqs.append(traj_id)
                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index]

    print('----------Loading data----------')
    train_dataset = TrajectoryDatasetTrain(pd.read_csv(args.data_train))
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=args.workers,
                              collate_fn=lambda x: x)

    # ========== 初始化模型
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        A = torch.from_numpy(A)
    X = X.to(device=args.device, dtype=torch.float)
    A = A.to(device=args.device, dtype=torch.float)
    args.gcn_nfeat = X.shape[1]
    poi_embed_model = GCN(ninput=args.gcn_nfeat,
                          nhid=args.gcn_nhid,
                          noutput=args.poi_embed_dim,
                          dropout=args.gcn_dropout)
    node_attn_model = NodeAttnMap(in_features=X.shape[1], nhid=args.node_attn_nhid, use_mask=False)
    num_users = len(user_id_dict)
    user_embed_model = UserEmbeddings(num_users, args.user_embed_dim)
    time_embed_model = Time2Vec(args.time_embed_dim)
    cat_embed_model = CategoryEmbeddings(num_cats, args.cat_embed_dim)
    embed_fuse_model1 = FuseEmbeddings(args.user_embed_dim, args.poi_embed_dim)
    embed_fuse_model2 = FuseEmbeddings(args.time_embed_dim, args.cat_embed_dim)
    args.seq_input_embed = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim + args.cat_embed_dim
    seq_model = TransformerModel(num_pois,
                                 num_cats,
                                 args.seq_input_embed,
                                 args.transformer_nhead,
                                 args.transformer_nhid,
                                 args.transformer_nlayers,
                                 dropout=args.transformer_dropout)
    optimizer = optim.Adam(params=list(poi_embed_model.parameters()) +
                                  list(node_attn_model.parameters()) +
                                  list(user_embed_model.parameters()) +
                                  list(time_embed_model.parameters()) +
                                  list(cat_embed_model.parameters()) +
                                  list(embed_fuse_model1.parameters()) +
                                  list(embed_fuse_model2.parameters()) +
                                  list(seq_model.parameters()),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_time = maksed_mse_loss
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor)

    def input_traj_to_embeddings(sample, poi_embeddings):
        # user day seq name poi
        # Parse sample
        input_seq = [each[0] for each in sample[1]]
        input_seq_time = [each[1] for each in sample[1]]
        input_seq_cat = [poi_cat_dict[each] for each in input_seq]
        # User to embedding
        user_id = sample[0]
        user_idx = user_id_dict[str(user_id)]
        input = torch.LongTensor([user_idx]).to(device=args.device)
        user_embedding = user_embed_model(input)
        user_embedding = torch.squeeze(user_embedding)
        # POI to embedding and fuse embeddings
        input_seq_embed = []
        for idx in range(len(input_seq)):
            poi_embedding = poi_embeddings[input_seq[idx]]
            poi_embedding = torch.squeeze(poi_embedding).to(device=args.device)
            # Time to vector
            time_embedding = time_embed_model(
                torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device=args.device))
            time_embedding = torch.squeeze(time_embedding).to(device=args.device)
            # Categroy to embedding
            cat_idx = torch.LongTensor([input_seq_cat[idx]]).to(device=args.device)
            cat_embedding = cat_embed_model(cat_idx)
            cat_embedding = torch.squeeze(cat_embedding)
            # Fuse user+poi embeds
            fused_embedding1 = embed_fuse_model1(user_embedding, poi_embedding)
            fused_embedding2 = embed_fuse_model2(time_embedding, cat_embedding)
            # Concat time, cat after user+poi
            concat_embedding = torch.cat((fused_embedding1, fused_embedding2), dim=-1)
            # Save final embed
            input_seq_embed.append(concat_embedding)
        return input_seq_embed

    def adjust_pred_prob_by_graph(y_pred_poi):
        y_pred_poi_adjusted = torch.zeros_like(y_pred_poi)
        attn_map = node_attn_model(X, A)
        for i in range(len(batch_seq_lens)):
            traj_i_input = batch_input_seqs[i]  # list of input check-in pois
            for j in range(len(traj_i_input)):
                y_pred_poi_adjusted[i, j, :] = attn_map[traj_i_input[j], :] + y_pred_poi[i, j, :]
        return y_pred_poi_adjusted

    # ========== 训练
    poi_embed_model = poi_embed_model.to(device=args.device)
    node_attn_model = node_attn_model.to(device=args.device)
    user_embed_model = user_embed_model.to(device=args.device)
    time_embed_model = time_embed_model.to(device=args.device)
    cat_embed_model = cat_embed_model.to(device=args.device)
    embed_fuse_model2 = embed_fuse_model2.to(device=args.device)
    seq_model = seq_model.to(device=args.device)

    train_epochs_top1_acc_list = []
    train_epochs_top5_acc_list = []
    train_epochs_top10_acc_list = []
    train_epochs_top20_acc_list = []
    train_epochs_mAP20_list = []
    train_epochs_mrr_list = []
    train_epochs_loss_list = []
    train_epochs_poi_loss_list = []
    train_epochs_time_loss_list = []
    train_epochs_cat_loss_list = []
    # For saving ckpt
    max_train_score = -np.inf

    for epoch in range(args.epochs):
        logging.info(f"{'*' * 50}Epoch:{epoch:03d}{'*' * 50}\n")
        poi_embed_model.train()
        node_attn_model.train()
        user_embed_model.train()
        time_embed_model.train()
        cat_embed_model.train()
        embed_fuse_model1.train()
        embed_fuse_model2.train()
        seq_model.train()

        train_batches_top1_acc_list = []
        train_batches_top5_acc_list = []
        train_batches_top10_acc_list = []
        train_batches_top20_acc_list = []
        train_batches_mAP20_list = []
        train_batches_mrr_list = []
        train_batches_loss_list = []
        train_batches_poi_loss_list = []
        train_batches_time_loss_list = []
        train_batches_cat_loss_list = []
        src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)
        # Loop batch
        for b_idx, batch in enumerate(train_loader):
            if len(batch) != args.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)
            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            batch_seq_labels_time = []
            batch_seq_labels_cat = []
            poi_embeddings = poi_embed_model(X, A)
            # Convert input seq to embeddings
            for sample in batch:
                # sample[0]: traj_id, sample[1]: input_seq, sample[2]: label_seq
                traj_id = sample[0]
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                input_seq_time = [each[1] for each in sample[1]]
                label_seq_time = [each[1] for each in sample[2]]
                label_seq_cats = [poi_cat_dict[each] for each in label_seq]
                input_seq_embed = torch.stack(input_traj_to_embeddings(sample, poi_embeddings))
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))

            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
            label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)

            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            y_pred_poi, y_pred_time, y_pred_cat = seq_model(x, src_mask)

            # Graph Attention adjusted prob
            y_pred_poi_adjusted = adjust_pred_prob_by_graph(y_pred_poi)

            loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)
            loss_time = criterion_time(torch.squeeze(y_pred_time), y_time)
            loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)

            # Final loss
            loss = loss_poi + loss_time * args.time_loss_weight + loss_cat
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi_adjusted.detach().cpu().numpy()
            batch_pred_times = y_pred_time.detach().cpu().numpy()
            batch_pred_cats = y_pred_cat.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
            train_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            train_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            train_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            train_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            train_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            train_batches_mrr_list.append(mrr / len(batch_label_pois))
            train_batches_loss_list.append(loss.detach().cpu().numpy())
            train_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            train_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
            train_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())

            # Report training progress
            if (b_idx % (args.batch * 5)) == 0:
                sample_idx = 0
                batch_pred_pois_wo_attn = y_pred_poi.detach().cpu().numpy()
                logging.info(f'Epoch:{epoch}, batch:{b_idx}, '
                             f'train_batch_loss:{loss.item():.2f}, '
                             f'train_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                             f'train_move_loss:{np.mean(train_batches_loss_list):.2f}\n'
                             f'train_move_poi_loss:{np.mean(train_batches_poi_loss_list):.2f}\n'
                             f'train_move_time_loss:{np.mean(train_batches_time_loss_list):.2f}\n'
                             f'train_move_top1_acc:{np.mean(train_batches_top1_acc_list):.4f}\n'
                             f'train_move_top5_acc:{np.mean(train_batches_top5_acc_list):.4f}\n'
                             f'train_move_top10_acc:{np.mean(train_batches_top10_acc_list):.4f}\n'
                             f'train_move_top20_acc:{np.mean(train_batches_top20_acc_list):.4f}\n'
                             f'train_move_mAP20:{np.mean(train_batches_mAP20_list):.4f}\n'
                             f'train_move_MRR:{np.mean(train_batches_mrr_list):.4f}\n'
                             f'traj_id:{batch[sample_idx][0]}\n'
                             f'input_seq: {batch[sample_idx][1]}\n'
                             f'label_seq:{batch[sample_idx][2]}\n'
                             f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_cat:{[poi_cat_dict[each[0]] for each in batch[sample_idx][2]]}\n'
                             f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_time:{list(batch_seq_labels_time[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n'
                             f'pred_seq_time:{list(np.squeeze(batch_pred_times)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                             '=' * 100)

        epoch_train_top1_acc = np.mean(train_batches_top1_acc_list)
        epoch_train_top5_acc = np.mean(train_batches_top5_acc_list)
        epoch_train_top10_acc = np.mean(train_batches_top10_acc_list)
        epoch_train_top20_acc = np.mean(train_batches_top20_acc_list)
        epoch_train_mAP20 = np.mean(train_batches_mAP20_list)
        epoch_train_mrr = np.mean(train_batches_mrr_list)
        epoch_train_loss = np.mean(train_batches_loss_list)
        epoch_train_poi_loss = np.mean(train_batches_poi_loss_list)
        epoch_train_time_loss = np.mean(train_batches_time_loss_list)
        epoch_train_cat_loss = np.mean(train_batches_cat_loss_list)

        train_epochs_loss_list.append(epoch_train_loss)
        train_epochs_poi_loss_list.append(epoch_train_poi_loss)
        train_epochs_time_loss_list.append(epoch_train_time_loss)
        train_epochs_cat_loss_list.append(epoch_train_cat_loss)
        train_epochs_top1_acc_list.append(epoch_train_top1_acc)
        train_epochs_top5_acc_list.append(epoch_train_top5_acc)
        train_epochs_top10_acc_list.append(epoch_train_top10_acc)
        train_epochs_top20_acc_list.append(epoch_train_top20_acc)
        train_epochs_mAP20_list.append(epoch_train_mAP20)
        train_epochs_mrr_list.append(epoch_train_mrr)

        monitor_loss = epoch_train_loss
        monitor_score = np.mean(epoch_train_top1_acc * 4 + epoch_train_top20_acc)

        # Learning rate schuduler
        lr_scheduler.step(monitor_loss)

        # Print epoch results
        logging.info(f"Epoch {epoch}/{args.epochs}\n"
                     f"train_loss:{epoch_train_loss:.4f}, "
                     f"train_poi_loss:{epoch_train_poi_loss:.4f}, "
                     f"train_time_loss:{epoch_train_time_loss:.4f}, "
                     f"train_cat_loss:{epoch_train_cat_loss:.4f}, "
                     f"train_top1_acc:{epoch_train_top1_acc:.4f}, "
                     f"train_top5_acc:{epoch_train_top5_acc:.4f}, "
                     f"train_top10_acc:{epoch_train_top10_acc:.4f}, "
                     f"train_top20_acc:{epoch_train_top20_acc:.4f}, "
                     f"train_mAP20:{epoch_train_mAP20:.4f}, "
                     f"train_mrr:{epoch_train_mrr:.4f}")

        state_dict = {
            'epoch': epoch,
            'poi_embed_state_dict': poi_embed_model.state_dict(),
            'node_attn_state_dict': node_attn_model.state_dict(),
            'user_embed_state_dict': user_embed_model.state_dict(),
            'time_embed_state_dict': time_embed_model.state_dict(),
            'cat_embed_state_dict': cat_embed_model.state_dict(),
            'embed_fuse1_state_dict': embed_fuse_model1.state_dict(),
            'embed_fuse2_state_dict': embed_fuse_model2.state_dict(),
            'seq_model_state_dict': seq_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'user_id_dict': user_id_dict,
            'poi_id_dict': poi_id_dict,
            'cat_id_dict': cat_id_dict,
            'poi_cat_dict': poi_cat_dict,
            'node_attn_map': node_attn_model(X, A),
            'args': args,
            'epoch_train_metrics': {
                'epoch_train_loss': epoch_train_loss,
                'epoch_train_poi_loss': epoch_train_poi_loss,
                'epoch_train_time_loss': epoch_train_time_loss,
                'epoch_train_cat_loss': epoch_train_cat_loss,
                'epoch_train_top1_acc': epoch_train_top1_acc,
                'epoch_train_top5_acc': epoch_train_top5_acc,
                'epoch_train_top10_acc': epoch_train_top10_acc,
                'epoch_train_top20_acc': epoch_train_top20_acc,
                'epoch_train_mAP20': epoch_train_mAP20,
                'epoch_train_mrr': epoch_train_mrr
            }
        }
        model_save_dir = os.path.join(args.save_dir, 'checkpoints')
        # Save best train score epoch
        if monitor_score >= max_train_score:
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            torch.save(state_dict, rf"{model_save_dir}/best_epoch.state.pt")
            with open(rf"{model_save_dir}/best_epoch.txt", 'w') as f:
                print(state_dict['epoch_train_metrics'], file=f)
            max_train_score = monitor_score

        with open(os.path.join(args.save_dir, 'metrics-train.txt'), "w") as f:
            print(f'train_epochs_loss_list={[float(f"{each:.4f}") for each in train_epochs_loss_list]}', file=f)
            print(f'train_epochs_poi_loss_list={[float(f"{each:.4f}") for each in train_epochs_poi_loss_list]}', file=f)
            print(f'train_epochs_time_loss_list={[float(f"{each:.4f}") for each in train_epochs_time_loss_list]}',
                  file=f)
            print(f'train_epochs_cat_loss_list={[float(f"{each:.4f}") for each in train_epochs_cat_loss_list]}', file=f)
            print(f'train_epochs_top1_acc_list={[float(f"{each:.4f}") for each in train_epochs_top1_acc_list]}', file=f)
            print(f'train_epochs_top5_acc_list={[float(f"{each:.4f}") for each in train_epochs_top5_acc_list]}', file=f)
            print(f'train_epochs_top10_acc_list={[float(f"{each:.4f}") for each in train_epochs_top10_acc_list]}',
                  file=f)
            print(f'train_epochs_top20_acc_list={[float(f"{each:.4f}") for each in train_epochs_top20_acc_list]}',
                  file=f)
            print(f'train_epochs_mAP20_list={[float(f"{each:.4f}") for each in train_epochs_mAP20_list]}', file=f)
            print(f'train_epochs_mrr_list={[float(f"{each:.4f}") for each in train_epochs_mrr_list]}', file=f)


def predict(args, cur_user, plan, constraint, expand_day):
    # ========== 加载模型
    raw_X = load_poi_features(args.data_node_feats)[:, 0:-2]
    num_pois = raw_X.shape[0]
    one_hot_encoder = OneHotEncoder()
    cat_list = list(raw_X[:, 2])
    one_hot_encoder.fit(list(map(lambda x: [x], cat_list)))
    one_hot_rlt = one_hot_encoder.transform(list(map(lambda x: [x], cat_list))).toarray()
    num_cats = one_hot_rlt.shape[-1]
    X = np.zeros((num_pois, raw_X.shape[-1] - 1 + num_cats), dtype=np.float32)
    X[:, 0] = raw_X[:, 0]
    X[:, 2:num_cats + 2] = one_hot_rlt  # 把one hot编码插入到原始数据的第三列
    X[:, num_cats + 2:] = raw_X[:, 3:]
    with open(os.path.join('./data/one-hot-encoder.pkl'), 'wb') as f:
        pickle.dump(one_hot_encoder, f)
    raw_A = load_matrix(args.data_adj_mtx)
    A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')
    poi_id_dict, cat_id_dict, poi_cat_dict, user_id_dict = initiate_dict()
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        A = torch.from_numpy(A)
    X = X.to(device=args.device, dtype=torch.float)
    A = A.to(device=args.device, dtype=torch.float)
    args.gcn_nfeat = X.shape[1]
    poi_embed_model = GCN(ninput=args.gcn_nfeat,
                          nhid=args.gcn_nhid,
                          noutput=args.poi_embed_dim,
                          dropout=args.gcn_dropout)
    node_attn_model = NodeAttnMap(in_features=X.shape[1], nhid=args.node_attn_nhid, use_mask=False)
    num_users = len(user_id_dict)
    user_embed_model = UserEmbeddings(num_users, args.user_embed_dim)
    time_embed_model = Time2Vec(args.time_embed_dim)
    cat_embed_model = CategoryEmbeddings(num_cats, args.cat_embed_dim)
    embed_fuse_model1 = FuseEmbeddings(args.user_embed_dim, args.poi_embed_dim)
    embed_fuse_model2 = FuseEmbeddings(args.time_embed_dim, args.cat_embed_dim)
    args.seq_input_embed = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim + args.cat_embed_dim
    seq_model = TransformerModel(num_pois,
                                 num_cats,
                                 args.seq_input_embed,
                                 args.transformer_nhead,
                                 args.transformer_nhid,
                                 args.transformer_nlayers,
                                 dropout=args.transformer_dropout)
    state_dict = torch.load(args.model_path)
    poi_embed_model.load_state_dict(state_dict['poi_embed_state_dict'])
    node_attn_model.load_state_dict(state_dict['node_attn_state_dict'])
    user_embed_model.load_state_dict(state_dict['user_embed_state_dict'])
    time_embed_model.load_state_dict(state_dict['time_embed_state_dict'])
    cat_embed_model.load_state_dict(state_dict['cat_embed_state_dict'])
    embed_fuse_model1.load_state_dict(state_dict['embed_fuse1_state_dict'])
    embed_fuse_model2.load_state_dict(state_dict['embed_fuse2_state_dict'])
    seq_model.load_state_dict(state_dict['seq_model_state_dict'])

    def input_traj_to_embeddings(sample, poi_embeddings):
        # user day seq name poi
        # Parse sample
        input_seq = [each[0] for each in sample[1]]
        input_seq_time = [each[1] for each in sample[1]]
        input_seq_cat = [poi_cat_dict[each] for each in input_seq]
        # User to embedding
        user_id = sample[0]
        user_idx = user_id_dict[str(user_id)]
        input = torch.LongTensor([user_idx]).to(device=args.device)
        user_embedding = user_embed_model(input)
        user_embedding = torch.squeeze(user_embedding)
        # POI to embedding and fuse embeddings
        input_seq_embed = []
        for idx in range(len(input_seq)):
            poi_embedding = poi_embeddings[input_seq[idx]]
            poi_embedding = torch.squeeze(poi_embedding).to(device=args.device)
            # Time to vector
            time_embedding = time_embed_model(
                torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device=args.device))
            time_embedding = torch.squeeze(time_embedding).to(device=args.device)
            # Categroy to embedding
            cat_idx = torch.LongTensor([input_seq_cat[idx]]).to(device=args.device)
            cat_embedding = cat_embed_model(cat_idx)
            cat_embedding = torch.squeeze(cat_embedding)
            # Fuse user+poi embeds
            fused_embedding1 = embed_fuse_model1(user_embedding, poi_embedding)
            fused_embedding2 = embed_fuse_model2(time_embedding, cat_embedding)
            # Concat time, cat after user+poi
            concat_embedding = torch.cat((fused_embedding1, fused_embedding2), dim=-1)
            # Save final embed
            input_seq_embed.append(concat_embedding)
        return input_seq_embed

    def adjust_pred_prob_by_graph(y_pred_poi):
        y_pred_poi_adjusted = torch.zeros_like(y_pred_poi)
        attn_map = node_attn_model(X, A)
        for i in range(len(batch_seq_lens)):
            traj_i_input = batch_input_seqs[i]  # list of input check-in pois
            for j in range(len(traj_i_input)):
                y_pred_poi_adjusted[i, j, :] = attn_map[traj_i_input[j], :] + y_pred_poi[i, j, :]
        return y_pred_poi_adjusted

    # ========== 预测
    poi_embed_model.eval()
    node_attn_model.eval()
    user_embed_model.eval()
    time_embed_model.eval()
    cat_embed_model.eval()
    embed_fuse_model1.eval()
    embed_fuse_model2.eval()
    seq_model.eval()

    batch_input_seqs = []
    batch_seq_lens = []
    batch_seq_embeds = []
    poi_embeddings = poi_embed_model(X, A)

    interest = load_matrix('./data/interest.csv')[:, 1]
    interest_sort = sorted(range(len(interest)), key=lambda k: interest[k], reverse=True)

    while True:
        big_position = set()
        for key in plan:
            for each in plan[key]:
                if raw_X[each[0]-10001][10]:
                    big_position.add(raw_X[each[0]-10001][10])

        # 对于全空的一天，先插入一个热门景点
        if expand_day:
            temp_len = len(plan)
            for each in interest_sort:
                # 没选过+所属地区没选过+下属地区没选过+白天访问
                if (each + 10001) not in constraint['select-spot'] and raw_X[each][10] not in constraint['select-spot'] and (each + 10001) not in big_position and raw_X[each][9] == 0:
                    constraint['select-spot'].append(each + 10001)
                    constraint['all-budget'] += raw_X[each][5]
                    plan[temp_len+1] = [[each + 10001, raw_X[each][1], 9, 9 + raw_X[each][8], raw_X[each][5]]]
                    if raw_X[each][10]:
                        big_position.add(raw_X[each][10])
                    break

        # 切割batch
        # plan: {day1: id, name, start_time, end_time}
        # sample: user, [(poi, time),()]
        batch = []
        sample = [cur_user]
        poi_time = []
        plan_poi = {}
        for key in plan:
            p = plan[key]
            plan_poi[key] = [0, []]   # {day1: 是否需要补充行程, [已有poi]}
            for each in p:
                plan_poi[key][1].append(each[0])
                poi_time.append((poi_id_dict[each[0]], each[2] * 2 / 48))
            sample.append(poi_time)
            if p[-1][-2] < 18:  # 结束时间小于18点
                plan_poi[key][0] = 1
                batch.append(sample)
            sample = [cur_user]
            poi_time = []

        # 预测下一个景点
        need = True
        while need:
            for sample in batch:
                input_seq = [each[0] for each in sample[1]]
                input_seq_embed = torch.stack(input_traj_to_embeddings(sample, poi_embeddings))
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            x = batch_padded.to(device=args.device, dtype=torch.float)
            src_mask = seq_model.generate_square_subsequent_mask(x.shape[0]).to(args.device)
            y_pred_poi, y_pred_time, y_pred_cat = seq_model(x, src_mask)
            y_pred_poi_adjusted = adjust_pred_prob_by_graph(y_pred_poi)     # need_day * poi * 313
            index = 0
            for key in plan_poi:
                if plan_poi[key][0] == 1:     # 如果需要补充行程
                    sorted_id = sorted(range(len(y_pred_poi_adjusted[index][-1])), key=lambda k: y_pred_poi_adjusted[index][-1][k])
                    for j in sorted_id:
                        # 如果该景点没有已选+该景点所在的地区没有已选（已经去过鼓浪屿，第二天不会再去日光岩）+白天晚上时间约束+预算满足
                        if (j + 10001) not in constraint['select-spot'] and ((j + 10001) not in big_position or
                                raw_X[plan[key][-1][0] - 10001][10] == (j + 10001)) and (
                                raw_X[j][10] not in constraint['select-spot'] or plan[key][-1][0] == raw_X[j][10] or
                                raw_X[plan[key][-1][0] - 10001][10] == raw_X[j][10]) and (
                                raw_X[j][5] + constraint['all-budget']) <= constraint['user-budget'] / 2:
                            # 如果前面是他属的小景点，且自己不是他的大景点，时间约束要从他属大景点开始计算
                            if raw_X[plan[key][-1][0]-10001][10] in constraint['select-spot'] and raw_X[j][10] != raw_X[plan[key][-1][0]-10001][10] and (j+10001) != raw_X[plan[key][-1][0]-10001][10]:
                                temp_index = 0
                                for i in range(len(plan[key])):
                                    if plan[key][i][0] == raw_X[plan[key][-1][0]-10001][10]:
                                        temp_index = i
                                        break
                                end_time = (plan[key][temp_index][-2] + 1 + raw_X[j][8])
                                if (end_time >= 16 and raw_X[j][9] == 0) or end_time > 21:
                                    continue
                                plan[key].append([j + 10001, raw_X[j][1], plan[key][temp_index][-2] + 1, plan[key][temp_index][-2] + 1 + raw_X[j][8], raw_X[j][5]])
                            # 如果前面是同属的大景点
                            elif plan[key][-1][0] == raw_X[j][10]:
                                end_time = plan[key][-1][2] + raw_X[j][8]
                                if (end_time >= 16 and raw_X[j][9] == 0) or end_time > 21:
                                    continue
                                plan[key].append([j + 10001, raw_X[j][1], plan[key][-1][2], plan[key][-1][2] + raw_X[j][8], raw_X[j][5]])
                            else:
                                end_time = plan[key][-1][-2] + 1 + raw_X[j][8]
                                if (end_time >= 16 and raw_X[j][9] == 0) or end_time > 21:
                                    continue
                                plan[key].append([j + 10001, raw_X[j][1], plan[key][-1][-2] + 1, plan[key][-1][-2] + 1 + raw_X[j][8], raw_X[j][5]])
                            plan_poi[key][1].append(j + 10001)
                            if raw_X[j][10]:
                                big_position.add(raw_X[j][10])
                            constraint['all-budget'] += raw_X[j][5]
                            constraint['select-spot'].append(j + 10001)
                            batch[index][1].append((poi_id_dict[j + 10001], (plan[key][-1][-2] + 1 + raw_X[j][8])*2/48))
                            if plan[key][-1][-2] >= 18:
                                plan_poi[key][0] = 0
                            break
                    index += 1
            need = False
            for key in plan_poi:
                if plan_poi[key][0] == 1:
                    need = True
                    break

        if len(plan) == int(constraint['user-time'] / 24):
            break

    return plan, constraint
