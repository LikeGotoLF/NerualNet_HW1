# coding: utf-8
import sys
sys.path.append('/data/honglifeng/mnist/data')  # 为了导入父目录的文件而进行的设定
import numpy as np
from mnist import load_mnist
from PIL import Image



def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)

img = x_train[0]
label = t_train[0]
print('显示的数字是:',label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
print(img.shape)  # (28, 28)
img_show(img)