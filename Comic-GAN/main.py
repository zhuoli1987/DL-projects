
from glob import glob
from models import GAN
from dataset import Dataset

import os
import models
import dataset
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataDir', required=True, help='path to dataset, only need the folder name')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=96, help='the height / width of the input image to network')
parser.add_argument('--zsize', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--epoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--alpha', type=float, default=0.2, help='alpha, default=0.2')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--showevery', type=int, default=10, help='print out losses per x iterations')
parser.add_argument('--printevery', type=int, default=100, help='generate images per x iterations')
parser.add_argument('--d_labelSmooth', type=float, default=0.0, help='for D, use soft label "1-labelSmooth" for real samples')
parser.add_argument('--outDir', default='checkpoints', help='folder to output images and model checkpoints')
parser.add_argument('--gaussain', action='store_true', help='if yes, then use gaussain random noise')
parser.add_argument('--minibatch', action='store_true', help='if yes, then use the minibatch')


# arg_list = [
#     '--dataDir', 'anime-faces',
#	  '--checkpointDir', 'checkpoints'
#     '--workers', '12',
#     '--batchSize', '64',
#     '--imageSize', '96',
#     '--zsize', '200',
#     '--epoch', '80',
#     '--lr', '0.0002',
#     '--beta1', '0.5',
#     '--outDir', 'results',
#     '--d_labelSmooth', '0.1', # 0.25 from imporved-GAN paper 
# ]

args = parser.parse_args()

try:
    os.makedirs(args.outDir)
except OSError:
    pass


# Hyper parameters
data_folder_path = os.getcwd() + '/' + args.dataDir
output_dir = args.outDir
image_size = args.imageSize
real_size = (image_size, image_size, 3)
z_size = (1, 1, args.zsize)
learning_rate = args.lr
batch_size = args.batchSize
epochs = args.epoch
alpha = args.alpha
beta1 = args.beta1
d_labelSmooth = args.d_labelSmooth

# Create the network
model = GAN(real_size, z_size, image_size, learning_rate, alpha=alpha, beta1=beta1, label_smooth=d_labelSmooth,minibatch=args.minibatch)

# Prepare the data
dataset = Dataset(glob(os.path.join(data_folder_path, '**/danbooru*.*'), recursive=True), image_width=image_size, image_height=image_size)
                  
# Training the network
losses, samples = models.train(model, z_size, dataset, epochs, batch_size, print_every=args.showevery, show_every=args.printevery, use_gaussain=args.gaussain)

