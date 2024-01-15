from train import train
import argparse

if __name__ == '__main__':
    # ========== 设置超参数
    parser = argparse.ArgumentParser(description="Run GETNext.")
    parser.add_argument('--data-adj-mtx', type=str, default='../data/traj_graph_A.csv', help='Graph adjacent path')
    parser.add_argument('--data-node-feats', type=str, default='../data/spot.csv', help='Graph node features path')
    parser.add_argument('--poi-embed-dim', type=int, default=128, help='POI embedding dimensions')
    parser.add_argument('--gcn-dropout', type=float, default=0.3, help='Dropout rate for gcn')
    parser.add_argument('--gcn-nhid', type=list, default=[32, 64], help='List of hidden dims for gcn layers')
    parser.add_argument('--node-attn-nhid', type=int, default=128, help='Node attn map hidden dimensions')
    parser.add_argument('--hidden-size', type=int, default=128, help='RNN hidden dimensions')
    parser.add_argument('--num-heads', type=int, default=6, help='RNN hidden dimensions')
    parser.add_argument('--cat-embed-dim', type=int, default=32, help='Category embedding dimensions')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    args = parser.parse_args()
    # ========== 训练
    train(args)
