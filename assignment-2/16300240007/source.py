import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
import numpy as np
import matplotlib.pyplot as plt
import string
from handout import Dataset
from handout import get_text_classification_datasets
from handout import get_linear_seperatable_2d_2c_dataset

min_count = 10
num_class = 4
batch_size = 32

class LSM:
	# LSM for binary classification
	# still no regularization
	def __init__(self):
		self.W = None

	def train(self, X, y):
		N = X.shape[0]
		X_norm = np.hstack((np.ones((N, 1)), X))
		self.W = (np.linalg.inv((X_norm.T).dot(X_norm))).dot(X_norm.T).dot(y)

	def predict(self, X):
		if(self.W is None):
			return 
		N = X.shape[0]
		X_norm = np.hstack((np.ones((N, 1)), X))
		pred = X_norm.dot(self.W)
		return pred >= 0

	def plot(self, plt):
		b, w1, w2 = self.W
		k1 = - w1 / w2
		k2 = - b / w2
		x = np.array([-1, 0, 1])
		y = k1 * x + k2
		plt.xlabel("x1")
		plt.ylabel("x2")
		plt.plot(x, y, c = 'r', label = 'LSM')
		return plt

class perceptron:
	# perceptron for binary classification (seem like it's only for binary ones emmm)
	def __init__(self):
		self.W = None
		self.losses = []

	def train(self, X, y, learning_rate = 0.1, num_iter = 100):
		N = X.shape[0]
		D = X.shape[1]
		X_norm = np.hstack((np.ones((N, 1)), X))
		y_norm = 2 * y - 1 # map into {-1, 1}
	
		# initialization for weight matrix
		if self.W is None:
			self.W = 0.001 * np.random.randn(D + 1)

		# training process
		for i in range(num_iter):
			score = X_norm.dot(self.W) 
			pred = score >= 0
			index = np.where(pred != y)[0]
			loss = - np.sum(score[index] * y_norm[index])
			self.losses.append(loss)
			# randomly choose one sample to do SGD
			if(len(index) != 0):
				rand_index = np.random.randint(len(index))
				self.W += learning_rate * X_norm[index[rand_index]] * y_norm[index[rand_index]]

		plt.figure(1)
		plt.title("perceptron loss")
		plt.xlabel("number iteration")
		plt.ylabel('loss')
		plt.plot(range(num_iter), self.losses)

	def predict(self, X):
		if(self.W is None):
			return 
		N = X.shape[0]
		X_norm = np.hstack((np.ones((N, 1)), X))
		pred = X_norm.dot(self.W)
		return pred >= 0

	def plot(self, plt):
		b, w1, w2 = self.W
		k1 = - w1 / w2
		k2 = - b / w2
		x = np.array([-1, 0, 1])
		y = k1 * x + k2
		plt.xlabel("x1")
		plt.ylabel("x2")
		plt.plot(x, y, c = 'b', label = 'perceptron')
		return plt

def to_one_hot(y, D):
	'''
	Arguments:
		y: vector of (N,)
		D: dimension of one hot vector
	'''
	N = y.shape[0]
	ans = np.zeros([N, D])
	ans[range(N), y.astype(int)] = 1
	return ans

def get_vocabulary(string_list):
	N = len(string_list)
	vocabulary = {}
	frequency = {}
	for i in range(N):
		# deal with useless chars
		string_list[i] = string_list[i].lower()
		for j in range(len(string.punctuation)):
			string_list[i] = string_list[i].replace(string.punctuation[j], '')
		for j in range(len(string.whitespace)):
			string_list[i] = string_list[i].replace(string.whitespace[j], ' ')
		item = string_list[i].split(' ')

		# put into vocabulary
		for j in range(len(item)):
			if(item[j] != ''):
				if(item[j] in frequency.keys()):
					frequency[item[j]] += 1
				else:
					frequency[item[j]] = 1
	
	# check frequency
	count = 0
	for key, value in frequency.items():
		if(value >= min_count):
			vocabulary[key] = count
			count += 1
	return vocabulary

def to_word_vec(string_list, vocabulary):
	N = len(string_list)
	D = len(vocabulary)
	text_vector = np.zeros((N, D))

	for i in range(N):
		string_list[i] = string_list[i].lower()
		for j in range(len(string.punctuation)):
			string_list[i] = string_list[i].replace(string.punctuation[j], '')
		for j in range(len(string.whitespace)):
			string_list[i] = string_list[i].replace(string.whitespace[j], ' ')
		item = string_list[i].split(' ')
		
		for j in range(len(item)):
			if(item[j] in vocabulary.keys()):
				text_vector[i][vocabulary[item[j]]] = 1
	return text_vector

def preprocess(text_train, text_test):
	v = get_vocabulary(text_train.data)
	train_vector = to_word_vec(text_train.data, v)
	test_vector = to_word_vec(text_test.data, v)
	return train_vector, test_vector

class softmax:
	# SVM multi-class classification
	def __init__(self):
		self.W = None
		self.losses = []

	def loss(self, W, X, y, reg = 0.1):
		N, D = X.shape
		dW = np.zeros_like(W)
		X = np.hstack((np.ones((N, 1)), X))
		scores = X.dot(W)

		correct_score = scores[range(N), y].reshape(-1, 1) # [N,]
		exp_sum = np.sum(np.exp(scores), axis = 1).reshape(-1, 1)
		loss = np.sum(np.log(exp_sum) - correct_score)
		loss = loss / N + 0.5 * reg * np.sum(W[1:] * W[1:])

		temp = np.exp(scores) / exp_sum
		temp[range(N), y] -= 1 # [N, C]
		dW = X.T.dot(temp) / N + reg * W
		dW[0, :] += reg * W[0, :]
		return loss, dW

	def train(self, X, y, learning_rate = 0.1, batch_size = 32, num_epoch = 300, reg = 0.1):
		N = X.shape[0]
		D = X.shape[1]
		X_norm = np.hstack((np.ones((N, 1)), X))
		
		if self.W is None:
			self.W = 0.001 * np.random.randn(D + 1, num_class)

		iter_epoch = N // batch_size
		num_iter = num_epoch * iter_epoch
		for i in range(num_iter):
			rand_index = np.random.choice(N, batch_size, replace=True)
			X_batch = X[rand_index]
			y_batch = y[rand_index]
			loss, dW = self.loss(self.W, X_batch, y_batch, reg)
			self.W -= learning_rate * dW
			if(i % iter_epoch == 0):
				self.losses.append(loss)

	def predict(self, X):
		if(self.W is None):
			return 
		N = X.shape[0]
		X_norm = np.hstack((np.ones((N, 1)), X))
		pred = np.argmax(X_norm.dot(self.W), axis = 1)
		return pred

	def accuracy(self, X, y):
		y_pred = self.predict(X)
		return np.mean(y_pred == y)

	def plot(self, plt):
		plt.title("softmax loss")
		plt.xlabel("number iteration")
		plt.ylabel('loss')
		plt.plot(range(len(self.losses)), self.losses)
		return plt

if __name__ == "__main__":
	d = get_linear_seperatable_2d_2c_dataset()
	train_set, test_set = d.split_dataset()
	
	# Part 1: least square model
	plt.figure(0)
	LSM = LSM()
	LSM.train(train_set.X, 2 * train_set.y - 1)
	y_pred = LSM.predict(test_set.X)
	print("LSM accuracy: ", test_set.acc(y_pred))
	d.plot(plt)
	LSM.plot(plt)

	# Part 2: perceptron
	perceptron = perceptron()
	perceptron.train(train_set.X, train_set.y)
	y_pred = perceptron.predict(test_set.X)
	print("perceptron accuracy: ", test_set.acc(y_pred))
	plt.figure(0)
	perceptron.plot(plt)
	plt.legend(loc="upper right")
	plt.show()
	
	# Part 3: text classification
	text_train, text_test = get_text_classification_datasets()
	train_vector, test_vector = preprocess(text_train, text_test)
	
	N = train_vector.shape[0]
	plt.figure("text_classification")
	for j, i in enumerate([1, batch_size, N]):
		Softmax = softmax()
		Softmax.train(train_vector, text_train.target, batch_size = i)
		print(Softmax.accuracy(test_vector, text_test.target))
		plt.subplot(3, 1, j + 1)
		Softmax.plot(plt)
	plt.show()
