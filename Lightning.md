# pytorch-lightning

本质上pytorch和pytorch-lightning是相同的，只不过pytorch需要自己造轮子（如model, dataloader, loss, train，test，checkpoint, save model等等都需要自己写），而pl 把这些模块都**结构化**了（类似keras）。

## 定义模型

```python
import lightning as L

class LitAutoEncoder(L.LightningModule):
    def __init__(self, Net):
        super().__init__()
        self.Net = Net

    def forward(self, x):
        return self.Net(x)
    
    def criterion(self, x, y):
        return F.cross_entropy(x, y)
    
	# *结构化定义train loop
    def training_step(self, batch):
        x, y = batch
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
```

**值得注意的是，在这里没有.cuda()或者.to(device)要写，Lightning自动处理了这些信息**

训练时

```python
# model
autoencoder = LitAutoEncoder(Net)

# train model
trainer = L.Trainer()
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
```

按照官方文档的说法，这段代码做了以下的事情：

```python
autoencoder = LitAutoEncoder(Net)
optimizer = autoencoder.configure_optimizers()

for batch_idx, batch in enumerate(train_loader):
    loss = autoencoder.training_step(batch, batch_idx)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

可以查看类Trainer的参数

![image-20240826203618616](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/img/image-20240826203618616.png)

其中函数fit的参数

![image-20240826203506697](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/img/image-20240826203506697.png)

## 定义test&valid loop

```python
class LitAutoEncoder(L.LightningModule):
        ...

    def test_step(self, batc):
        x, y = batch
        x_hat = self.Net(z)
        test_loss = self.criterion(x_hat, y)
    
    def validation_step(self, batch):
        x, y = batch
        x_hat = self.Net(x)
        val_loss = self.criterion(x_hat, y)
```

### **测试时**

```python
autoencoder = LitAutoEncoder(Net)
trainer = L.Trainer()

trainer.test(model = autoencoder, dataloader = test_dataloader)
```

### **验证时**

> 官方文档的分割数据集方式值得学习：
>
> ```python
> import torch.utils.data as data
> from torchvision import datasets
> import torchvision.transforms as transforms
> 
> # use 20% of training data for validation
> train_set_size = int(len(train_set) * 0.8)
> valid_set_size = len(train_set) - train_set_size
> 
> # split the train set into two
> seed = torch.Generator().manual_seed(42)
> train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)
> ```
>
> 

```python
trainer.fit(model = autoencoder, train_loader, valid_loader)
```

## 保存&加载checkpoint

> 根据官方文档
>
> lightning框架的checkpoint有以下内容
>
> ![image-20240826204952229](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/img/image-20240826204952229.png)
>
> - 16位缩放因子（如果使用16位精度训练）
> - 当前时代
> - 全局step  ==（？）==
> - Lightning模块的state_dict
> - 所有优化器的状态
> - 所有学习率调度的状态
> - 所有回调的状态（对于有状态回调）==（？）==
> - 数据模块的状态（用于有状态的数据模块）
> - 创建模型的超参数（初始化参数）
> - 创建数据模块的超参数（init参数）
> - 循环状态

### 保存

默认状态下，Lightning会每个epoch保存一次checkpoint到当前目录下

当然可以通过Trainer类中的参数default_root_dir来定义默认的根目录，即默认目录

```python
# saves checkpoints to 'some/path/' at every epoch end
trainer = Trainer(default_root_dir="some/path/")
```



### 加载

```python
model = autoencoder.load_from_checkpoint("/path/to/checkpoint.ckpt")

# disable randomness, dropout, etc...
model.eval()

# predict with the model
y_hat = model(x)
```

这种加载方式仅仅加载其weights和超参，并不会加载优化器、epoch等等

如果需要不仅仅加载权重，而是回复整个训练的话可以直接用fit

```python
model = LitModel()
trainer = Trainer()

# automatically restores model, epoch, step, LR schedulers, etc...
trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")
```

**值得注意的是，Lightning框架的checkpoints完全兼容普通的pytorch**

即直接用torch.load是可以加载Lightning框架保存的checkpoints的

## 训练日志

```python
# *在LightningModule中的训练和验证循环模块中加入self.log()
	def training_step(self, batch):
        x, y = batch
        x = x.type(torch.float32)
        y = y.type(torch.float32)
        x_hat = self.forward(x)
        loss = F.cross_entropy(x_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch):
        x, y = val_batch
        x = x.type(torch.float32)
        y = y.type(torch.float32)
        x_hat = self.forward(x)
        loss = F.cross_entropy(x_hat, y)
        self.log('val_loss', loss)
```

训练结束后可以使用tensorboard进行查看

```python
>> tensorboard --logdir=lightning_logs/
```

## 疑问

> 1. 绝大多数的博客都是用包**pytorch_lightning**，而官方文档用的是**Lightning**
> 2. pytorch_lightning库中的trainer有gpus这个参数，但Lightning库中的trainer没有
> 3. 使用lightning框架时模型的初始化？

