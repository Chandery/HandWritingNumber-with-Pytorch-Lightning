import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L

class CCNet(nn.Module):
    def __init__(self):
        super(CCNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.MaxPool1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 18, kernel_size=3, padding=1)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.relu3 = nn.ReLU()

        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(288,10)
        self.relu4 = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1) # *label.shape = (bs, classnumber) 所以要保证所有类别加起来为1，因此dim=1

    def forward(self, x):
        x = self.relu1(self.MaxPool1(self.conv1(x)))
        
        x = self.relu2(self.MaxPool2(self.conv2(x)))

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.relu3(self.MaxPool3(x))

        x = self.dropout(x)
        x = self.flatten(x)
        x = self.relu4(self.fc(x))

        x = self.softmax(x)

        return x
    
    

class LitAutoEncoder(L.LightningModule):
    def __init__(self, Net, lr):
        super().__init__()
        self.Net = Net
        self.lr = lr

    def forward(self, x):
        x = self.Net.forward(x)

        return x

    def training_step(self, batch):
        x, y = batch
        x = x.type(torch.float32)
        y = y.type(torch.float32)
        x_hat = self.forward(x)
        loss = F.cross_entropy(x_hat, y)
        acc = y.argmax(dim=1).eq(x_hat.argmax(dim=1)).sum().item()/len(y)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss
    
    def validation_step(self, val_batch):
        x, y = val_batch
        x = x.type(torch.float32)
        y = y.type(torch.float32)
        x_hat = self.forward(x)
        loss = F.cross_entropy(x_hat, y)
        acc = y.argmax(dim=1).eq(x_hat.argmax(dim=1)).sum().item()/len(y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
if __name__ == '__main__':
    Net = CCNet()
    LitNet = LitAutoEncoder(Net)
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(LitNet)