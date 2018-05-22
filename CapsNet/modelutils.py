from keras import layers, initializers
from keras import backend as bk 
import tensorflow as tf 
import utils

class Mask(layers.Layer):
	"""
	Mask a tensor with shape=[None, d1, d2] by the max value in axis=1.
	ouput shape: [None, d1*d2]
	"""
	def call(self, inputs, **kwargs):
		# use true label to select target capsule, shape=[batch_size, num_capsule]
		if type(inputs) is list:
			assert len(inputs) == 2
			inputs, mask = inputs
		else: # if no true label, mask by the max length of vectors of capsules
			# compute the length of the capsules
			x = bk.sqrt(bk.sum(bk.square(inputs), -1))
			# one-hot encoded
			# mask.shape=[None, n_classes]=[None, num_capsule]
			mask = bk.one_hot(indices=bk.argmax(x, 1), num_classes=x.get_shape().as_list()[1])
		
		# inputs.shape=[None, num_capsule, dim_vector]
		# mask.shape=[None, num_capsule]
		# masked.shape=[None, num_capsule * dim_vector]
		inputs_masked = bk.batch_flatten(inputs * bk.expand_dims(mask, -1))
		return inputs_masked

	def compute_output_shape(self, input_shape):
		if type(input_shape[0]) is tuple:  # true label provided
			return tuple([None, input_shape[0][1] * input_shape[0][2]])
		else:  # no true label provided
			return tuple([None, input_shape[1] * input_shape[2]])

	def get_config(self):
		config = super(Mask, self).get_config()

class Length(layers.Layer):

	def call(self, inputs, **kwargs):
		# L2 length which is the square root
		# of the sum of square of the capsule element
		return bk.sqrt(bk.sum(bk.square(inputs), -1))

	def compute_output_shape(self, input_shape):
		return input_shape[:-1]

	def get_config(self):
		config = super(Length, self).get_config()
		return config

def PrimaryCap(inputs, dim_vector, n_channels, kernel_size=5, strides=1, padding=0):
	"""
	Apply 2D convolutions to generate n_channel capsules with dimension of dim_vector
	:param:inputs: 4D tensor, shape=[None, width, height, channels]
	:param:dim_vector: the dimension of the output vector of capsules
	:param:n_channels: the number of capsules
	:return:output: tensor, shape=[None, num_capsule, dim_vector] 
	"""
	output = layers.Conv2D(filters=dim_vector*n_channels, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
	output = layers.Reshape(target_shape=[-1, dim_vector])(output)
	return layers.Lambda(utils.squash)(output)


class DigiCap(layers.Layer):

	def __init__(self, num_capsule, dim_vector, num_routing=3, 
				kernel_initializer='glorot_uniform',
				b_initializer='zeros',
				**kwargs):
		super(DigiCap, self).__init__(**kwargs)

		self.num_capsule = num_capsule # 10 for Mnist
		self.dim_vector = dim_vector # 16
		self.num_routing = num_routing # 3
		self.kernel_initializer = initializers.get(kernel_initializer)
		self.b_initializer = initializers.get(b_initializer)

	def build(self, input_shape):
		"""
		"""
		assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
		self.input_num_capsule = input_shape[1]
		self.input_dim_vector = input_shape[2]

		# Transform matrix W
		self.W = self.add_weight(shape=[self.input_num_capsule,
										self.num_capsule, 
										self.input_dim_vector,
										self.dim_vector],
								initializer=self.kernel_initializer,
								name='W')

		# Coupling coefficient
		self.b = self.add_weight(shape=[1,
										self.input_num_capsule,
										self.num_capsule,
										1,
										1],
								initializer=self.b_initializer,
								name='b',
								trainable=False)

		self.built = True

	def call(self, inputs, training=None):
		# inputs.shape = (None, input_num_capsule, input_dim_vector)
		# -> (None, input_num_capsule, 1, 1, input_dim_vector)
		inputs_expand = bk.expand_dims(bk.expand_dims(inputs, 2), 2)

		# replicate num_capsule dimension to prepare being multiplied by W
		# -> [None, input_num_capsule, num_capsule, 1, input_dim_vector]
		inputs_tiled = bk.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])


		# Compute `inputs * W` by scanning inputs_tiled on dimension 0. 
        # -> [None, input_num_capsule, num_capsule, 1, dim_vector]
		inputs_hat = tf.scan(lambda ac, x: bk.batch_dot(x, self.W, [3, 2]),
							 elems=inputs_tiled,
							 initializer=bk.zeros([self.input_num_capsule,
							 					   self.num_capsule,
							 					   1,
							 					   self.dim_vector]))

		# Routing algorithm
		assert self.num_routing > 0, 'The num_routing should be > 0.'
		for i in range(self.num_routing):
			c = tf.nn.softmax(self.b, dim=2) # dim=2 is the num_capsule dimension

			# [None, 1, num_capsule, 1, dim_vector]
			outputs = utils.squash(bk.sum(c * inputs_hat, 1, keepdims=True))

			# The last iteration needs not compute b which will not be passed to the graph
			if i != self.num_routing - 1:
				self.b += bk.sum(inputs_hat * outputs, -1, keepdims=True)

		return bk.reshape(outputs, [-1, self.num_capsule, self.dim_vector])

	def compute_output_shape(self, input_shape):
		return tuple([None, self.num_capsule, self.dim_vector])

	def get_config(self):
		config = {
			'num_capsule': self.num_capsule,
			'dim_vector': self.dim_vector,
			'num_routing': self.num_routing
		}
		base_config = super(DigiCap, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
