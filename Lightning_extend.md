# Pytorch-Lightning补充和扩展

## Lighting是什么

> Pytorch-Lightning是一个轻量级的Pytorch深度学习框架，旨在**简化和规范**深度学习模型的训练过程。它的好处在于提供了一组模块和接口，使用户能够更容易地组织和训练模型。同时减少样板代码的数量。PyTorch-Lightning的设计目标是**提高代码的可读性和可维护性**，同时保持灵活性。它通过将训练循环的组件**拆分**为独立的模块（如模型、优化器、调度器等），以及提供默认实现来简化用户代码。这使得用户可以专注于模型的定义和高级训练策略，而不必处理底层训练循环的细节。

通过简介，可以了解Lightning是基于Pytorch的一个框架，最大的特点就是对代码进行拆分，称为几个独立的模块，包装起来，使得编码和训练过程更加简化和规范。

<img src="https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/img/cc67f3bf4ae2acc144aa28cf79ae9f8f.jpg" alt="Getting started with pytorch lightning - getting started"  />

要理解Lighting框架的基本构造，需要理解Lightning框架在构建模型的时候是构建一个模型系统，即不仅仅像Pytorch继承nn.Module的类，仅仅建立模型的定义，而是在建立模型基本结构定义的基础上，加入一系列模块的定义，包括优化器、训练循环、验证循环等等，继承pl.LightningModule的类，使得后续训练、验证非常的便捷

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

值得注意的是，在这个类中是不需要对device进行设置的，即.cuda()或者.to(device)等等Pytorch的写法，Lightning框架中封装了这些信息。

除此之外还可以对模型的test，valid的循环进行定义，可以自定义准确率之类的指标

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
        # acc = y.argmax(dim=1).eq(x_hat.argmax(dim=1)).sum().item()/len(y)
```



## 训练

对以上模型系统进行封装之后，训练变得很方便

只需要使用Lightning框架中的Trainer()类，使用方法fit即可

```python
# model
autoencoder = LitAutoEncoder(Net)

# train model
trainer = L.Trainer()
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
```

这个代码在框架之下实际上相当于如下代码

```python
autoencoder = LitAutoEncoder(Net)
optimizer = autoencoder.configure_optimizers()

for batch_idx, batch in enumerate(train_loader):
    loss = autoencoder.training_step(batch, batch_idx)

    loss.backward()  
    optimizer.step()
    optimizer.zero_grad()
```

可以看到，我们在模型系统模块中定义的training_step其实就是每次放入train_loader中的一个batch，然后进行自定义的操作，Lighting封装好了对loss进行的反向传播，然后用定义好的优化器进行优化

## Debugging

> 使用Lightning框架之后，第一个遇到的问题就是Debug的问题。因为运行的过程被封装在了框架内部，因此调试起来很困难，同时又因为框架训练过程集成了进度条，输出调试也有了限制，无法每个epoch或者batch都进行输出，否则就会和进度条的屏幕刷新冲突，出现闪屏，根本看不清楚等等。
>
> 为解决这个问题，Lightning框架集成了不少用于debug的功能函数或者参数，方便使用者进行调试。

**这里分享几个减少训练体量以快速找出bug的功能**

### Trainer的fast_dev_run参数

```python
trainer = Trainer(fast_dev_run=True)
```

这个参数表示在这个设置下运行程序，程序只会进行一个batch的训练，一个batch的验证和一个batch的测试和预测。

方便我们输出调试也好、单步调试也好，对代码逻辑进行debug

当然这个参数也可以设置称为数量n，表示跑n个batch

当然值得注意的是，这个参数当且仅当debug的时候使用，会禁用一些功能

### 数据规模的减小参数

在很多调试情况下我们不希望使用全部数据进行训练,只需要确定程序是否按照预期路线执行,Lightning给了一个这样的调试参数

```python
# use only 10% of training data and 1% of val data
trainer = Trainer(limit_train_batches=0.1, limit_val_batches=0.01)

# use 10 batches of train and 5 batches of val
trainer = Trainer(limit_train_batches=10, limit_val_batches=5)
```

这个方法也是很实用的,可以用参数设置使得数据规模展示减小,方便调试

### 输出模型每层的维度

这个在模型建立的阶段是非常实用的!可以通过输出,动态的调整模型的超参,使得模型更加合理

设置的方法也很简单,只需要在LightningModule初始化的方法中加入example_input_array即可

```python
class LitModel(LightningModule):
    def __init__(self, *args, **kwargs):
        self.example_input_array = torch.Tensor(32, 1, 28, 28)
```

```python
  | Name  | Type        | Params | Mode  | In sizes  | Out sizes
----------------------------------------------------------------------
0 | net   | Sequential  | 132 K  | train | [10, 256] | [10, 512]
1 | net.0 | Linear      | 131 K  | train | [10, 256] | [10, 512]
2 | net.1 | BatchNorm1d | 1.0 K  | train | [10, 512] | [10, 512]
```

### 如何使用单步调试对Lightning框架写的代码进行调试

对于上述输出调试无法有效进行的时候,经常用到单步调试对代码一步一步运行,查看每个变量有没有问题.

但是和未封装的pytorch不同,Lightning的运行循环是被封装起来的,因此无法和熟悉的循环一样单步运行调试

经过探索,总结了以下方法（以Vscode为例）

1.在模型模块的方法定义中的定义训练循环的地方打上断点

![image 20240909211052747](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/img/image-20240909211052747.png)

2.运行程序即可进入training_step的断点处，这时候再进行单步运行就可以一步一步查看模型运行的结果和loss的变化；

3.需要看多个batch的时候，每次loop结束让程序继续运行而非单步运行，即可再次停在training_step的断点处，重复此过程即可。

## 结果展示

![image 20240909213650980](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/img/image-20240909213650980.png)

![image 20240909213702638](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/img/image-20240909213702638.png)

![image 20240909213713901](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/img/image-20240909213713901.png)

![image 20240909213721740](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/img/image-20240909213721740.png)