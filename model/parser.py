import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(description="Run GETNext.")
    parser.add_argument('--device', type=str, default=torch.device('cpu'), help='')
    # Data
    parser.add_argument('--data-adj-mtx', type=str, default='./data/cost.csv', help='Graph adjacent path')
    parser.add_argument('--data-node-feats', type=str, default='./data/spot.csv', help='Graph node features path')
    parser.add_argument('--data-train', type=str, default='./data/traj.csv', help='Training data path')
    parser.add_argument('--short-traj-thres', type=int, default=2, help='Remove over-short trajectory')
    # Model
    parser.add_argument('--poi-embed-dim', type=int, default=128, help='POI embedding dimensions')
    parser.add_argument('--user-embed-dim', type=int, default=128, help='User embedding dimensions')
    parser.add_argument('--time-embed-dim', type=int, default=32, help='Time embedding dimensions')
    parser.add_argument('--gcn-dropout', type=float, default=0.3, help='Dropout rate for gcn')
    parser.add_argument('--gcn-nhid', type=list, default=[32, 64], help='List of hidden dims for gcn layers')
    parser.add_argument('--node-attn-nhid', type=int, default=128, help='Node attn map hidden dimensions')
    parser.add_argument('--hidden-size', type=int, default=128, help='RNN hidden dimensions')
    parser.add_argument('--num-heads', type=int, default=6, help='RNN hidden dimensions')
    parser.add_argument('--cat-embed-dim', type=int, default=32, help='Category embedding dimensions')
    parser.add_argument('--transformer-nhid', type=int, default=1024, help='Hid dim in TransformerEncoder')
    parser.add_argument('--transformer-nlayers', type=int, default=2, help='Num of TransformerEncoderLayer')
    parser.add_argument('--transformer-nhead', type=int, default=2, help='Num of heads in multiheadattention')
    parser.add_argument('--transformer-dropout', type=float, default=0.3, help='Dropout rate for transformer')
    # Train
    parser.add_argument('--time-loss-weight', type=int, default=10, help='Scale factor for the time loss term')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--batch', type=int, default=20, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--lr-scheduler-factor', type=float, default=0.1, help='Learning rate scheduler factor')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    # Experiment config
    parser.add_argument('--workers', type=int, default=0, help='Num of workers for dataloader.')
    parser.add_argument('--project', default='../output', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--model-path', type=str, default='./output/exp-3/checkpoints/best_epoch.state.pt', help='Model path.')

    args = parser.parse_args(args=[])
    return args
