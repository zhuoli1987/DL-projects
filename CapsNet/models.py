import numpy as np
import matplotlib.pyplot as plt

from keras import backend as bk
from keras import layers, models, optimizers, callbacks
from keras.preprocessing.image import ImageDataGenerator
from modelutils import PrimaryCap, DigiCap, Length, Mask
from utils import plot_log, combine_images
from PIL import Image

M_PLUS = 0.9
M_MINUS = 0.1
LAMBDA = 0.5

bk.set_image_data_format('channels_last')

def margin_loss(y_true, y_pred):
	"""
	:param: y_true: [None, n_classes]
	:param: y_pred: [None, nm_capsule]
	:return: a scalar loss value
	"""
	L = y_true * bk.square(bk.maximum(0., M_PLUS - y_pred)) + \
		LAMBDA * (1 - y_true) * bk.square (bk.maximum(0., y_pred - M_MINUS))

	return bk.mean(bk.sum(L, 1))

def train_generator(x, y, batch_size, shift_fraction=0.):
	train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
									   height_shift_range=shift_fraction) # shift up to 2 pixel for MINIST

	generator = train_datagen.flow(x, y, batch_size=batch_size)
	while 1:
		x_batch, y_batch = generator.next()
		yield([x_batch, y_batch], [y_batch, x_batch])


def CapsNet(input_shape, n_class, num_routing):

	# Image input
	x = layers.Input(shape=input_shape)

	# ReLU Conv1
	conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

	# Primary Capsules: Conv2D layer, 'squash' activation
	# then reshape to [None, num_capsule, dim_vector]
	primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

	# Digi Capsules: The routing algorithm works here
	digicaps = DigiCap(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='digitcaps')(primarycaps)

	# This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary.
	out_caps = Length(name='out_caps')(digicaps)

	# Decoder network
	y = layers.Input(shape=(n_class,))

	# The true label is used to extract the corresponding vj
	masked_y = Mask()([digicaps, y])
	masked = Mask()(digicaps)

	# shared decoder model in training nd prediction
	decoder = models.Sequential(name='decoder')
	decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
	decoder.add(layers.Dense(1024, activation='relu'))
	decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
	decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

	# Models for training and evaluation
	train_model = models.Model([x, y], [out_caps, decoder(masked_y)])
	eval_model = models.Model(x, [out_caps, decoder(masked)])

	return train_model, eval_model
	

def train(model, data, args):
	"""
 	Training a CapsuleNet
	:param model: the CapsuleNet model
	:param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
	:param args: arguments
	:return: The trained model
	"""
	# unpack the data
	(x_train, y_train), (x_test, y_test) = data

	# callbacks
	log = callbacks.CSVLogger(args.save_dir + '/log.csv')
	tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
							   batch_size=args.batch_size,
							   histogram_freq=int(args.debug))
	checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:0.2d}.h5', 
										   monitor='val_capsnet_acc',
    									   save_best_only=True,
    									   save_weights_only=True,
    									   verbose=1)
	lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
	model.compile(optimizer=optimizers.Adam(lr=args.lr),
    			  loss=[margin_loss, 'mse'],
    			  loss_weights=[1., args.lam_recon],
    			  metrics={'capsnet' : 'accuracy'})


	# Training with data augmentation
	model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
    					steps_per_epoch=int(y_train.shape[0] / args.batch_size),
    					epochs=args.epochs,
    					validation_data=[[x_test, y_test],[y_test, x_test]],
    					callbacks=[log, tb, checkpoint, lr_decay])

    # Save trained model
	model.save_weights(args.save_dir + '/trained_model.h5')
	print('Trained model saved to\'{}/trained_model.h5\''.format(args.save_dir))

	plot_log(args.save_dir + '/log.csv', show=True)

	return model


def test(model, data, args):
	x_test, y_test = data
	y_pred, x_recon = model.predict(x_test, batch_size=100)
	print('-'*30 + 'Begin: test' + '-'*30)
	print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

	img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
	image = img * 255
	Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/real_and_recon.png')
	print()
	print('Reconstructed images are saved to {}/real_and_recon.png'.format(args.save_dir))
	print('-'*30 + 'End: test' + '-'*30)
	plt.imshow(ply.imread(args.save_dir + '/real_and_recon.png'))
	plt.show()
