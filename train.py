import torch
from dataloader import NumberDataset
import argparse
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('-m',"--model", type=str, default="CCNet", help="Model to train")
    parser.add_argument('-l',"--init_lr", type=float, default=2e-4,help="Initial learning rate")
    parser.add_argument('-b',"--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument('-e',"--epochs", type=int, default=50, help="Number of epochs")
    return parser.parse_args()

def train(args):
    # *取出命令行中的参数
    init_lr = args.init_lr
    bs = args.batch_size
    epochs = args.epochs
    model_name = args.model
    # *根据参数选择模

    # *设置训练集、验证集、测试集数据的路径
    img_path_train = "archive/train_imgs.npy"
    label_path_train = "archive/train_labels.npy"
    # *根据数据划分不同的数据集并封装成Dataloader类
    train_dataset = NumberDataset(img_path_train, label_path_train, "train")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=False)

    valid_dataset = NumberDataset(img_path_train, label_path_train, "val")
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=bs, shuffle=False)

    
    from Net import LitAutoEncoder as LAE
    from Net import CCNet
    import lightning as L

    trainer = L.Trainer(max_epochs=epochs, precision=32)

    Net = CCNet()

    model = LAE(Net, init_lr)

    trainer.fit(model, train_dataloader, valid_dataloader)

if __name__ == '__main__':
    args = parse_args()
    # print(args)
    train(args)