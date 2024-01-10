import pickle

import train
import argparse

if __name__ == '__main__':
    # ========== 设置超参数
    args = argparse.ArgumentParser(description="Run GETNext.")
    args.add_argument('--poi-embed-dim', type=int, default=128, help='POI embedding dimensions')
    args.add_argument('--gcn-dropout', type=float, default=0.3, help='Dropout rate for gcn')
    args.add_argument('--gcn-nhid', type=list, default=[32, 64], help='List of hidden dims for gcn layers')
    args.add_argument('--node-attn-nhid', type=int, default=128, help='Node attn map hidden dimensions')
    # ========== 训练
    train(args)
