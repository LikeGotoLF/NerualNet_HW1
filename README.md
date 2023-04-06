# NerualNet_HW1
构建两层神经网络分类器

数据集为mnist，请自行下载并解压缩至NerualNet_HW1/data下，在mnist.py和model.py文件内修改路径。

```
#训练模式
python model.py train
#测试模式
python model.py test
```

###训练部分
激活函数使用ReLU和softmax，损失使用交叉熵损失、随机梯度下降法、L2正则化、学习率衰减
最终参数如下：
|  参数   | 数值  |  备注|
|  ----  | ----  |----  |
| lr  | 0.1 |学习率 |
| hidden_size  | 300 | 隐藏层 |
| alpha  | 0.001 | 正则化参数 |
| lr_decay_rate  | 0.99 | 学习率衰减 |
