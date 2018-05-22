import os
import argparse
import numpy as np

from models import CapsNet, train
from datasets import load_mnist

parser = argparse.ArgumentParser(description='Capsule Network on MNIST.')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs to train for')
parser.add_argument('--batch_size', default=100, type=int, help='input batch size')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate, default=0.001')
parser.add_argument('--lr_decay', default=0.9, type=float, help='The value multiplied by lr at each epoch. Set a larger value for larger epochs')
parser.add_argument('--lam_recon', default=0.392, type=float, help='THe coefficient for the loss of decoder')
parser.add_argument('-r', '--routings', default=3, type=int,
					help='Number of iterations used in routing algorithm. should be > 0')
parser.add_argument('--shift_fraction', default=0.1, type=float, 
					help='Fraction of pixels to shift at most in each direction.')
parser.add_argument('--debug', action='store_true', help='Save weights by TensorBoard')
parser.add_argument('--save_dir', default='./result')
parser.add_argument('-t', '--testing', action='store_true',
					help='Test the trained model on testing dataset')
parser.add_argument('--digit', default=5, type=int, help='Digit to manipulate')
parser.add_argument('-w', '--weights', default=None, help='The path of the save weights. Should be specified when testing')

args = parser.parse_args()

if not os.path.exists(args.save_dir):
	os.makedirs(args.save_dir)

# load data
(x_train, y_train), (x_test, y_test) = load_mnist()

# define model
model, eval_model = CapsNet(input_shape=x_train.shape[1:], n_class=len(np.unique(np.argmax(y_train, 1))), num_routing=args.routings)
model.summary()

# train or test
if args.weights is not None: # init the model weights if there is already saved ones
	model.load_weights(args.weights)

if not args.testing:
	train(model=model, data=((x_train, y_train),(x_test, y_test)), args=args)
else:
	if args.weights is None:
		print('Random initialized weights will be tested')

	test(model=eval_model, data=(x_test, y_test), args=args)
