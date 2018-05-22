from keras.datasets import mnist
from keras.utils import to_categorical

def load_mnist():
	# the data shuffled and split between train and test sets
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
	x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
	y_train = to_categorical(y_train.astype('float32'))
	y_test = to_categorical(y_test.astype('float32'))

	return (x_train, y_train), (x_test, y_test)