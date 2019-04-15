import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
from handout import GaussianMixture1D
import numpy as np
import matplotlib.pyplot as plt

def kde(data, x, h = 0.2):
	N = len(data)
	density = np.zeros_like(x)
	for i in range(N):
		density += np.exp(-np.square(x - data[i]) / (2 * np.square(h)))
	density /= N * np.sqrt(2 * np.pi) * h
	return density

def KNN(data, x, K = 20):
	N = len(data)
	data_np = np.array((data)).reshape((1, -1))
	data_np = np.tile(data_np,(len(x),1))
	x_np = x.reshape((-1, 1))
	dist = np.abs(data_np - x_np)
	dist.sort()
	return K / (N * dist[:, K])

def cross_validation(data, k, h):
	# K折交叉验证
	s = 0
	N = len(data)
	batch_size = N // k
	data_batch = [data[i:i + batch_size] for i in range(0, N, batch_size)]
	for i in range(5):
		train = []
		for j in range(k):
			if(j != i):
				train.extend(data_batch[j])
		# 最大似然估计
		s += sum(np.log(kde(train, np.array(data_batch[i]), h)))
	return s / k

def find_h(data):
	hs = np.linspace(0.1, 1, 1000)
	k = 5
	h_best = 0
	s_max = -10000
	s_history = []
	for i in range(1000):
		s = cross_validation(data, k, hs[i])
		s_history.append(s)
		if(s > s_max):
			s_max = s
			h_best = hs[i]
	plt.title("log likelihood for h")
	plt.plot(hs, s_history)
	return h_best

if __name__ == "__main__":
	# preprocess
	np.random.seed(0)
	gm1d = GaussianMixture1D(mode_range=(0, 50))
	sampled_data = gm1d.sample([10000])
	sampled_data = get_data(10000)
	x = np.linspace(20, 40, 2000)

	# Part 1: talk about the influence of data size
	data_size = [100, 500, 1000, 10000]
	plt.figure("hist")
	for i in range(len(data_size)):
		plt.subplot(2, 2, i + 1)
		plt.title("num_data = " + str(data_size[i]))
		plt.hist(sampled_data[:data_size[i]], normed=True, bins=50)

	plt.figure("kernel_density_estimation")
	for i in range(len(data_size)):
		plt.subplot(2, 2, i + 1)
		plt.title("num_data = " + str(data_size[i]))
		plt.plot(x, kde(sampled_data[:data_size[i]], x))

	plt.figure("KNN_density_estimation")
	for i in range(len(data_size)):
		plt.subplot(2, 2, i + 1)
		plt.title("num_data = " + str(data_size[i]))
		plt.plot(x, KNN(sampled_data[:data_size[i]], x))
	plt.show()

	# Part 2: talk about the influence of hist bins
	sampled_data = sampled_data[:200]
	plt.figure(1)
	plt.subplot(1,3,1)
	plt.title("bins=5")
	plt.hist(sampled_data, normed=True, bins=5)
	plt.subplot(1,3,2)
	plt.title("bins=20")
	plt.hist(sampled_data, normed=True, bins=20)
	plt.subplot(1,3,3)
	plt.title("bins=50")
	plt.hist(sampled_data, normed=True, bins=50)

	plt.figure(2)
	plt.subplot(1,3,1)
	plt.title("Square-root choice bins=15")
	plt.hist(sampled_data, normed=True, bins=15)
	plt.subplot(1,3,2)
	plt.title("Sturges' formula bins=9")
	plt.hist(sampled_data, normed=True, bins=9)
	plt.subplot(1,3,3)
	plt.title("Rice Rule bins=12")
	plt.hist(sampled_data, normed=True, bins=12)

	plt.show()
	# Part 3: find best h	 
	sample = sampled_data[:100]
	h = find_h(sample)
	plt.figure(3)
	plt.title('kde when h = ' + str(h))
	plt.plot(x, kde(sample, x, h))
	plt.show()
	
	# Part 4: compare different k
	Ks = [1, 5, 30, 50]
	plt.figure(4)
	for i in range(4):
		plt.subplot(2, 2, i + 1)
		plt.title("K = " + str(Ks[i]))
		plt.plot(x, KNN(sampled_data, x, Ks[i]))
		gm1d.plot(num_sample=100)
		plt.ylim(0, 1)
	plt.show()