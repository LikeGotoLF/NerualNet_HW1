import sys
import random
import numpy as np
import seaborn as sn
from mnist import load_mnist
import matplotlib.pyplot as plt
sys.path.append('/data/honglifeng/mnist/data')  
class TwoLayerNet(object):
    def __init__(self,input_size=784,hidden_size=100,output_size=10,weight_init_std=0.01,lr=1e-3,alpha=1e-3):
        self.lr=lr
        self.alpha=alpha
        self.types=output_size
        self.params={}
        self.params['W1']=weight_init_std*np.random.randn(input_size,hidden_size)#正态分布初始化,但存在方差过大or过小的问题
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2']=weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['b2']=np.zeros(output_size)
        self.result={}
        self.result['h1']=None
        self.result['relu']=None
        self.result['h2']=None
        self.result['out']=None
        self.gradient={}
        self.gradient['W2']=None
        self.gradient['b2']=None
        self.gradient['W1']=None
        self.gradient['b1']=None
    def forward(self,X):
        self.result['h1']=X@self.params['W1']+self.params['b1']
        self.result['relu']=self.ReLU(self.result['h1'])
        self.result['h2']=self.result['relu']@self.params['W2']+self.params['b2']     
        self.result['out']=self.softmax(self.result['h2'])
        return self.result['out']
    def ReLU(self,X):
        return np.maximum(0, X)
    def softmax(self,h):
        exp_h = np.exp(h - np.max(h, axis=-1, keepdims=True))
        sm = exp_h/np.sum(exp_h, axis=-1, keepdims=True)
        return sm
    def CrossEntrophy(self,out,y,epsilon=1e-12):
        out = np.clip(out, epsilon, 1.-epsilon)  #防止溢出
        info=-np.log(out)
        return np.mean(y*info)
    def one_hot(self,y):
        encode=np.eye(self.types)[y]
        return encode
    def backward(self,X,out,y,penalty=None):
        batch_num=X.shape[0]
        dL_dz=(out-y)/batch_num#这里求平均了，所以下面求和就好
        self.gradient['W2']=self.result['relu'].T@dL_dz
        self.gradient['b2']=np.sum(dL_dz,axis=0)
        
        dL_dh1=self.params['W2']@dL_dz.T
        dL_dh1[(self.result['h1']).T <= 0] = 0 
        self.gradient['W1']=(dL_dh1@X).T
        self.gradient['b1']=np.sum(dL_dh1.T,axis=0)
        if penalty=='l2':
            self.gradient['W1']+=self.alpha*np.mean(self.params['W1'])
            self.gradient['W2']+=self.alpha*np.mean(self.params['W2'])
    def step(self):
        for item in self.params.keys():
            self.params[item]-=self.lr*self.gradient[item]
    def save_model(self,path):
        to_save={
            'model_name':'TwoLayerNet',
            'input_size':self.params['W1'].shape[0],
            'hidden_size':self.params['W1'].shape[1],
            'output_size':self.params['W2'].shape[1],
            'learning_rate':self.lr,
            'L2_rate':self.alpha,
            'params':self.params
        }
        np.save(path,to_save)
    def load_model(self,path):
        to_fit=np.load(path,allow_pickle=True).tolist()
        self.lr=to_fit['learning_rate']
        self.alpha=to_fit['L2_rate']
        for item in self.params.keys():
            self.params[item]=to_fit['params'][item]
            
def train_test(epoch,batch_size,lr,net,lr_decay_rate=0.99):
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    x_train=x_train
    x_test=x_test
    idx=list(range(x_train.shape[0]))
    random.shuffle(idx)
    train_loss,train_acc=[],[]
    test_loss,test_acc=[],[]
    for e in range(epoch):
        loss_e=[]
        acc_e=[]
        for i in range(0,x_train.shape[0],batch_size) :
            X=x_train[idx[i:i+batch_size]]
            y=net.one_hot(t_train[idx[i:i+batch_size]])
            out=net.forward(X)
            loss_e.append(net.CrossEntrophy(out,y))
            acc_e.append(calc_acc(out,y))
            net.backward(X,out,y,'l2')
            net.step()
        lr*=lr_decay_rate
        
        train_loss.append(np.mean(loss_e))
        train_acc.append(np.mean(acc_e))
        #----------------------------------test
        test_out=net.forward(x_test)
        test_y=net.one_hot(t_test)
        test_loss.append(net.CrossEntrophy(test_out,test_y))
        test_acc.append(calc_acc(test_out,test_y))
        print(f'---------epoch {e+1}---------')
        print(f'train loss {train_loss[-1]}')
        print(f'test loss {test_loss[-1]}')
        print(f'train acc {train_acc[-1]}')
        print(f'test acc {test_acc[-1]}')
    draw(train_loss,test_loss,train_acc,test_acc,net.params)
    net.save_model('/data/honglifeng/mnist/TLN.npy')     
    return test_acc[-1]             
def calc_acc(out,y):
    labels=np.argmax(y,axis=1)
    preds=np.argmax(out,axis=1)
    acc=(sum(labels==preds))/len(labels)
    return acc

def draw(train_loss,test_loss,train_acc,test_acc,params):
    fig=sn.heatmap(params['W1'])
    fig.get_figure().savefig('./W1.jpg')
    sn.heatmap(params['W2'])
    fig.get_figure().savefig('./W2.jpg')
    x_list=np.arange(len(train_loss))
    train_loss = [float(item) for item in train_loss]
    test_loss = [float(item) for item in test_loss]
    train_acc = [float(item) for item in train_acc]
    test_acc = [float(item) for item in test_acc]
    plt.figure(figsize=(20, 10), dpi=70)  # 设置图像大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplot(1,2,1)
    plt.plot(x_list, train_loss, color="lightcoral", linewidth=5.0, linestyle="-", label="train loss")
    plt.plot(x_list, test_loss, color="mediumpurple", linewidth=5.0, linestyle="--", label="test loss")
    plt.legend(["train loss", "test loss"], ncol=2)
    plt.xlabel("epoch",fontsize = 20)
    plt.ylabel("Loss",fontsize = 20)
    plt.subplot(1,2,2)
    plt.plot(x_list, train_acc ,color="lightcoral", linewidth=5.0, linestyle="-", label="train acc")
    plt.plot(x_list, test_acc, color="mediumpurple", linewidth=5.0, linestyle="--", label="test acc")
    plt.legend(["train acc", "test acc"], ncol=2)
    plt.xlabel("epoch",fontsize = 20)
    plt.ylabel("ACC",fontsize = 20)
    plt.savefig('./loss_acc.jpg')
     
def para_find(x_test,t_test):
    lrs =[0.1,0.01,0.001,0.0001]
    hidden_sizes=[100,200,300]
    alphas=[0.1,0.01,0.001,0.0001]
    best_acc=float('-inf')
    best_params={'lr':None,'hidden':None,'alpha':None}
    for lr in lrs:
        for hidden in hidden_sizes:
            for alpha in alphas:
                 net=TwoLayerNet(hidden_size=hidden,lr=lr,alpha=alpha)   
                 test_acc=train_test(130,256,lr,net)
                 if test_acc>best_acc:
                     best_acc=test_acc
                     best_params['lr']=lr
                     best_params['hidden']=hidden
                     best_params['alpha']=alpha 
    return best_params
                 
if __name__ == '__main__':
    
    args = sys.argv
    if args[1]=='train':
        net=TwoLayerNet(784,300,10,0.01)
        train_test(130,256,1e-1,net)
    elif args[1]=='test':
        net=TwoLayerNet()
        net.load_model('/data/honglifeng/mnist/TLN.npy')
        (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
        y_test=net.one_hot(t_test)
        out=net.forward(x_test)
        print(f'test accuracy:{calc_acc(out,y_test)}')
