
from PIL import Image
import numpy as np
import os

def get_image(image_path, width, height, mode):
    """
    Read image from image_path
    :param image_path: Path of image
    :param width: Width of image
    :param height: Height of image
    :param mode: Mode of image
    :return: Image data
    """
    image = Image.open(image_path)
    
    if image.size != (width, height):
        # Resize image
        image = image.resize((width, height))
    
    return np.array(image.convert(mode))

def get_batch(image_files, width, height, mode):
    data_batch = np.array(
        [get_image(file, width, height, mode) for file in image_files]).astype(np.float32)
    
    # Make sure the images are in 4 dimensions
    if len(data_batch.shape) < 4:
        data_batch = data_batch.reshape(data_batch.shape + (1,))
        
    return data_batch

def scale(x, feature_range=(-1, 1)):
    # scale to (0, 1)
    x = ((x - x.min())/(255 - x.min()))
    
    # scale to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x

def view_samples(epoch, samples, nrows, ncols, figsize=(5,5)):
    """
    Helper method to visualize the generated outout
    :param epoch: the number of iteration
    :param samples: generated image array
    """
    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, 
                             sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        ax.axis('off')
        img = ((img - img.min())*255 / (img.max() - img.min())).astype(np.uint8)
        ax.set_adjustable('box-forced')
        im = ax.imshow(img, aspect='equal')
    
    # No gap between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig, axes
 
def save_samples(epoch, samples, ouput_dir):
    """
    Helper method to save the generated output
    :param epoch: the number of iteration
    :param samples: generated image array
    :param outputdir: dir path to store all the images
    """
    try:
        os.makedirs(ouput_dir)
    except OSError:
        pass

    print(len(samples))
    index=1
    for img in samples:
        img = ((img - img.min())*255 / (img.max() - img.min())).astype(np.uint8)
        img = Image.fromarray(img)
        print(type(img))
        img.save(ouput_dir + '/gen_img_{}_{}.jpg'.format(epoch,index))
        index += 1
