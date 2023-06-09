# NerualNet_HW1
Numpy构建两层神经网络分类器，手写反向传播，loss以及梯度的计算。

数据集为mnist，请自行下载并解压缩至NerualNet_HW1/data下，在mnist.py和model.py文件内修改路径。

除了加载数据代码，所有实现均在model.py文件内。
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
| weight_init_std  | 0.01 |权重初始化 |
| epoch  | 130 | 周期 |

##### 训练结果可视化：
损失与准确率变化：
![loss and acc](https://github.com/LikeGotoLF/NerualNet_HW1/blob/main/loss_acc.jpg)

权重可视化：

![W1](https://github.com/LikeGotoLF/NerualNet_HW1/blob/main/W1.jpg)

![W2](https://github.com/LikeGotoLF/NerualNet_HW1/blob/main/W2.jpg)

### 测试部分
##### 模型文件百度云链接：链接: https://pan.baidu.com/s/1_FxGpt108flABy3bdjhNcA?pwd=yzyg 提取码: yzyg
##### 最终模型对测试集的预测准确率达到98.03%。


### 参数查找
para_find函数
