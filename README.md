# NerualNet_HW1
构建两层神经网络分类器

数据集为mnist，请自行下载并解压缩至NerualNet_HW1/data下，在mnist.py和model.py文件内修改路径。

```
#训练模式
python model.py train
#测试模式
python model.py test
```

### 训练部分
激活函数使用ReLU和softmax，损失使用交叉熵损失、随机梯度下降法、L2正则化、学习率衰减
最终参数如下：
|  参数   | 数值  |  备注|
|  ----  | ----  |----  |
| lr  | 0.1 |学习率 |
| hidden_size  | 300 | 隐藏层 |
| alpha  | 0.001 | 正则化参数 |
| lr_decay_rate  | 0.99 | 学习率衰减 |
| weight_init_std  | 0.99 |权重初始化 |
| epoch  | 130 | 周期 |

##### 训练结果可视化：
损失与准确率变化：
![loss and acc](https://github.com/LikeGotoLF/NerualNet_HW1/blob/main/loss_acc.jpg)

权重可视化：

W1表现了图像更底层的信息

![W1](https://github.com/LikeGotoLF/NerualNet_HW1/blob/main/W1.jpg)

W2表现了图像较高层的信息，开始出现一些小单位结构

![W2](https://github.com/LikeGotoLF/NerualNet_HW1/blob/main/W2.jpg)
