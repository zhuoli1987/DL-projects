
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint 
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np

# define function to load train, test and validation datasets
def load_dog_dataset(path):
	data = load_files(path)
	dog_files = np.array(data['filenames'])
	dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
	return dog_files, dog_targets

def Resnet50_model_train():

	# Load train, test, and validation datasets
	train_files, train_targets = load_dog_dataset('dogImages/train')
	valid_files, valid_targets = load_dog_dataset('dogImages/valid')
	test_files, test_targets = load_dog_dataset('dogImages/test')

	# Obtain bottleneck features from another pre-trained CNN.
	bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
	train_Resnet50 = bottleneck_features['train']
	valid_Resnet50 = bottleneck_features['valid']
	test_Resnet50 = bottleneck_features['test']

	# Transfer learning
	Resnet50_model = Sequential()
	Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
	Resnet50_model.add(Dense(133, activation='softmax'))

	# Compile the model.
	Resnet50_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

	# Train the model.
	checkpointer = ModelCheckpoint(filepath='model_weights/weights.best.Resnet50.hdf5', 
    	verbose=0, save_best_only=True)

	epochs=20
	Resnet50_model.fit(train_Resnet50, train_targets, validation_data=(valid_Resnet50, valid_targets), epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=0)