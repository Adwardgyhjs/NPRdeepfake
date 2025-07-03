import numpy as np
import matplotlib.pyplot as plt

# 定义x的范围
x = np.linspace(-10, 10, 400)

# 定义各个激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, x * alpha)

def tanh(x):
    return np.tanh(x)

def maxout(x, w1, b1, w2, b2):
    return np.maximum(w1 * x + b1, w2 * x + b2)

def relu(x):
    return np.maximum(0, x)

def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

# 计算各个函数的值
y_sigmoid = sigmoid(x)
y_leaky_relu = leaky_relu(x)
y_tanh = tanh(x)
y_maxout = maxout(x, np.array([1]), np.array([0]), np.array([0.5]), np.array([1]))
y_relu = relu(x)
y_elu = elu(x)

# 创建画布和子图
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

# 绘制各个函数在不同子图中
axs[0, 0].plot(x, sigmoid(x))
axs[0, 0].set_title('Sigmoid',fontsize=20)
axs[0, 0].text(-8, 0.8, r'$\sigma(x) = \frac{1}{1 + e^{-x}}$', fontsize=20)


axs[0, 1].plot(x, leaky_relu(x))
axs[0, 1].set_title('Leaky ReLU',fontsize=20)
axs[0, 1].text(-8, 8, r'$\text{Leaky ReLU}(x) = max(0.1x,x)$', fontsize=20)


axs[0, 2].plot(x, tanh(x))
axs[0, 2].set_title('Tanh',fontsize=20)
axs[0, 2].text(-8, 0.8, r'$\tanh(x)$', fontsize=20)

axs[1, 0].plot(x, maxout(x[:, np.newaxis], np.array([1]), 0, np.array([0.5]), 1))
axs[1, 0].set_title('Maxout',fontsize=20)
axs[1, 0].text(-8, 0.8, r'$\max(w_1^T x + b_1, w_2^T x + b_2)$', fontsize=20)

axs[1, 1].plot(x, relu(x))
axs[1, 1].set_title('ReLU',fontsize=20)
axs[1, 1].text(-8, 8, r'$\text{ReLU}(x) = \max(0, x)$', fontsize=20)

axs[1, 2].plot(x, elu(x))
axs[1, 2].set_title('ELU',fontsize=20)
axs[1, 2].text(-8, 0.8, r'$\text{ELU}(x) = {x ,x<0 ;a(e^x-1),x>0} $', fontsize=20)

# 调整子图之间的间距
plt.tight_layout()
plt.savefig("test1111.svg")
plt.show()
