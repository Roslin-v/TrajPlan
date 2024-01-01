from model import  *
from train import  *


if __name__ == '__main__':
    # ========== 设置超参数
    args = 0
    # ========== 初始化模型
    model = TPN()
    # ========== 训练
    train(model, args)
