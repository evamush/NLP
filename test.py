import tensorflow as tf

import numpy as np

import sklearn.preprocessing as prep

from tensorflow.examples.tutorials.mnist import input_data

# fan_in输入节点数量 fan_out输出节点数量

# 让权重被初始化得不大不小，正好合适

# mean = 0; variance = 2/(fan_in + fan_out)

def xavier_init(fan_in, fan_out, constant=1):

low = -constant * np.sqrt(6.0 / (fan_in + fan_out))

high = constant * np.sqrt(6.0 / (fan_in + fan_out))

return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):

# 输入变量数 隐层节点数 激活函数 优化器 scale高斯噪声系数

def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,

optimizer=tf.train.AdamOptimizer(), scale=0.1):

self.n_input = n_input

self.n_hidden = n_hidden

self.transfer = transfer_function

self.scale = tf.placeholder(tf.float32)

self.training_scale = scale

# 参数初始化, 初始化函数_initialize_weights

networks_weights = self._initialize_weights()

self.weights = networks_weights

# --- 开始定义网络结构 ---

self.x = tf.placeholder(tf.float32, [None, self.n_input])

# 建立提取特征的隐含层

# 1 先将输入x加入噪声，再将加了噪声的输入与隐含层的权重w1相乘，并加上隐含层的偏置b1

# 2 对结果进行激活函数处理

self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),

self.weights['w1']), self.weights['b1']))

# 经过隐含层后，需要在输出层进行数据复原、重建操作

self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

# 定义自编码的损失函数, 平方误差

self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))

# 定义训练操作的优化器

self.optimizer = optimizer.minimize(self.cost)

# 创建Session

init = tf.global_variables_initializer()

self.sess = tf.Session()

self.sess.run(init)

# --- 定义初始化函数_initialize_weights ---

def _initialize_weights(self):

# 创建字典

all_weights = dict()

# w1初始化

all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))

# 其它初始化为0

all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))

all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))

all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))

return all_weights

# 定义一个batch数据进行训练并返回当前cost

def partial_fit(self, X):

# feed_dict为输入数据X和噪声系数

cost, opt = self.sess.run((self.cost, self.optimizer),

feed_dict={self.x: X, self.scale: self.training_scale})

return cost

# 定义一个只计算cost的函数

def calc_total_cost(self, X):

return self.sess.run(self.cost,

feed_dict={self.x: X, self.scale: self.training_scale})

# 返回自编码器隐含层的输出结果:提取的特征

def transform(self, X):

return self.sess.run(self.hidden,

feed_dict={self.x: X, self.scale: self.training_scale})

# 定义函数进行单独重建:隐含层输出

def generate(self, hidden=None):

if hidden is None:

hidden = np.random.normal(size=self.weights['b1'])

return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

# 整体运行复原过程：提取高阶特征+通过高阶特征复原数据

def reconstruct(self, X):

return self.sess.run(self.resctruction,

feed_dict={self.x: X, self.scale: self.training_scale})

# 获取隐含层的权重

def getWeights(self):

return self.sess.run(self.weights['w1'])

# 获取隐含层的偏置系数

def getBiases(self):

return self.sess.run(self.weights['b1'])

# 载入MINIST数据集

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 对训练和测试数据进行标准化处理。让数据变成0均值，且标准差为1的分布:先减去均值，在除以标准差

def standard_scale(X_train, X_test):

preprocessor = prep.StandardScaler().fit(X_train)

X_train = preprocessor.transform(X_train)

X_test = preprocessor.transform(X_test)

return X_train, X_test

# 定义一个获取随机block数据的函数：不放回抽样

def get_random_block_from_data(data, batch_size):

# 随机整数

start_index = np.random.randint(0, len(data) - batch_size)

return data[start_index:(start_index + batch_size)]

# 对训练集、测试集进行标准化变换

X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

# 总训练样本数

n_samples = int(mnist.train.num_examples)

# 最大训练轮数

training_epochs = 20

batch_size = 128

display_step = 1

# 创建一个AGN自编码器

autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784,

n_hidden=200,

transfer_function=tf.nn.softplus,

optimizer=tf.train.AdagradOptimizer(learning_rate=0.001),

scale=0.01)

# 开始训练

for epoch in range(training_epochs):

# 平均损失函数

avg_cost = 0.

# 总共的batch数

total_batch = int(n_samples / batch_size)

for i in range(total_batch):

# 使用get_random_block_from_data获取随机batch数据

batch_xs = get_random_block_from_data(X_train, batch_size)

# 使用partial_fit进行训练，并返回cost

cost = autoencoder.partial_fit(batch_xs)

# 将当前的cost整合到avg_cost中

avg_cost += cost / n_samples * batch_size

# 每一轮迭代后，显示当前的迭代数和这一轮迭代的平均cost

if epoch % display_step == 0:

print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

# 对训练完的模型进行性能测试

print("Total Cost: " + str(autoencoder.calc_total_cost(X_test)))

作者：阿成9
链接：http://www.jianshu.com/p/a5d125dc64df
來源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。