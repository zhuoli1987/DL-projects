from keras import backend as bk 
from matplotlib import pyplot as plt

import numpy as np
import csv
import math

def squash(vectors, axis=-1):
	"""
	The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
	like a sigmoid
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
	"""
	s_squared_norm = bk.sum(bk.square(vectors), axis, keepdims=True)
	scale = s_squared_norm / (1 + s_squared_norm) / bk.sqrt(s_squared_norm + bk.epsilon())
	return scale * vectors

def plot_log(filename, show=True):
	keys=[]
	values=[]
	with open(filename, 'r') as file:
		reader = csv.DirectReader(file)
		for row in reader:
			if keys ==[]:
				for key, value in row.items():
					keys.append(key)
					values.append(value)
			else:
				for _, value in row.items():
					values.append(value)


		values = np.reshape(values, newshape=(-1, len(keys)))
		values[:,0] += 1 # not sure what this is for

		fig = plt.figure(figsize=(4, 6))
		fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
		fig.add_subplot(211)
		for i, key in enumerate(keys):
			if key.find('loss') >= 0 and not key.find('val') >= 0:
				plt.plot(values[:, 0], values[:, i], label=key)
		plt.legend()
		plt.title('Training Loss')

		fig.add_subplot(212)
		for i, key in enumerate(keys):
			if key.find('acc') >= 0:
				plt.plot(values[:, 0], values[:, i], label=key)
		plt.legend()
		plt.title('Training and Validation Accuracy')

		if show:
			plt.show()

def combine_images(generated_images, height=None, width=None):
	if width is None and height is None:
		width = int(math.sqrt(num))
		height = int(math.ceil(float(num)/width))
	elif width is not None and height is None:  # height not given
		height = int(math.ceil(float(num)/width))
	elif height is not None and width is None:  # width not given
		width = int(math.ceil(float(num)/height))

	shape = generated_images.shape[1:3]
	image = np.zeros((height*shape[0], width*shape[1]), dtype=generated_images.dtype)
	for index, img in enumerate(generated_images):
		i = int(index/width)
		j = index % width
		image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[:, :, 0]

	return image


