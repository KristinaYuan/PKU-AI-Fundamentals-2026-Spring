# -*- coding: utf-8 -*-
"""
@ author: Yiliang Liu
"""


# 作业内容：更改loss函数、网络结构、激活函数，完成训练MLP网络识别手写数字MNIST数据集

import numpy as np

from tqdm  import tqdm


# 加载数据集,numpy格式
X_train = np.load('./mnist/X_train.npy') # (60000, 784), 数值在0.0~1.0之间
y_train = np.load('./mnist/y_train.npy') # (60000, )
y_train = np.eye(10)[y_train] # (60000, 10), one-hot编码

X_val = np.load('./mnist/X_val.npy') # (10000, 784), 数值在0.0~1.0之间
y_val = np.load('./mnist/y_val.npy') # (10000,)
y_val = np.eye(10)[y_val] # (10000, 10), one-hot编码

X_test = np.load('./mnist/X_test.npy') # (10000, 784), 数值在0.0~1.0之间
y_test = np.load('./mnist/y_test.npy') # (10000,)
y_test = np.eye(10)[y_test] # (10000, 10), one-hot编码


# 定义激活函数
def relu(x):
    '''
    relu函数
    '''
    return np.maximum(0, x)

def relu_prime(x):
    '''
    relu函数的导数
    '''
    return (x > 0).astype(float)

def sigmoid(x):
    '''
    sigmoid函数
    '''
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    '''
    sigmoid函数的导数
    '''
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    '''
    tanh函数
    '''
    return np.tanh(x)

def tanh_prime(x):
    '''
    tanh函数的导数
    '''
    return 1.0 - np.tanh(x) ** 2

#输出层激活函数
def f(x):
    '''
    softmax函数, 防止除0
    '''
    x_shift = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def f_prime(x):
    '''
    softmax函数的导数 (不需要在此处单独使用，因为常与交叉熵结合计算)
    '''
    return np.ones_like(x)

# 定义损失函数
def loss_fn(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

def loss_fn_prime(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    return (y_pred - y_true) / y_true.shape[0]



# 定义权重初始化函数
def init_weights(shape=()):
    '''
    初始化权重
    '''
    return np.random.normal(loc=0.0, scale=np.sqrt(2.0/shape[0]), size=shape)

# 定义网络结构
class Network(object):
    '''
    MNIST数据集分类网络
    '''
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, lr=0.01):
        '''
        初始化网络结构
        '''
        self.W1 = init_weights((input_size, hidden_size1))
        self.b1 = np.zeros((1, hidden_size1))
        self.W2 = init_weights((hidden_size1, hidden_size2))
        self.b2 = np.zeros((1, hidden_size2))
        self.W3 = init_weights((hidden_size2, output_size))
        self.b3 = np.zeros((1, output_size))
        self.lr = lr

        # 用于保存中间变量供反向传播使用
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        self.z3 = None
        self.a3 = None

    def forward(self, x):
        '''
        前向传播
        '''
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = relu(self.z1)
        # self.a1 = tanh(self.z1)
        # self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # self.a2 = relu(self.z2)
        self.a2 = tanh(self.z2)
        # self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = f(self.z3) # Softmax
        return self.a3

    def step(self, x_batch, y_batch):
        '''
        一步训练
        '''
        # 前向传播
        out = self.forward(x_batch)

        # 计算损失和准确率
        loss = loss_fn(y_batch, out)
        acc = np.mean(np.argmax(out, axis=1) == np.argmax(y_batch, axis=1))

        # 反向传播
        dz3 = loss_fn_prime(y_batch, out)
        dW3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = np.dot(dz3, self.W3.T)
        # dz2 = da2 * relu_prime(self.z2)
        dz2 = da2 * tanh_prime(self.z2)
        # dz2 = da2 * sigmoid_prime(self.z2)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * relu_prime(self.z1)
        # dz1 = da1 * tanh_prime(self.z1)
        # dz1 = da1 * sigmoid_prime(self.z1)
        dW1 = np.dot(x_batch.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # 更新权重
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        
        return loss, acc

if __name__ == '__main__':
    # 训练网络
    net = Network(input_size=784, hidden_size1=256, hidden_size2=256, output_size=10, lr=0.01)
    batch_size = 64
    for epoch in range(10):
        losses = []
        accuracies = []
        p_bar = tqdm(range(0, len(X_train), batch_size))
        
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        for i in p_bar:
            x_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            loss, acc = net.step(x_batch, y_batch)
            losses.append(loss)
            accuracies.append(acc)
            p_bar.set_description(f"Epoch {epoch+1} Loss: {np.mean(losses):.4f}, Acc: {np.mean(accuracies):.4f}")
        
        val_out = net.forward(X_val)
        val_acc = np.mean(np.argmax(val_out, axis=1) == np.argmax(y_val, axis=1))
        print(f"Epoch {epoch+1} Val Acc: {val_acc:.4f}")
    
    test_out = net.forward(X_test)
    test_acc = np.mean(np.argmax(test_out, axis=1) == np.argmax(y_test, axis=1))
    print(f"Final Test Accuracy: {test_acc:.4f}")
