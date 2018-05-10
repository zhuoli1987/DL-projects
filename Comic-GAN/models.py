import pickle as pkl

import numpy as np
import tensorflow as tf
import modelutils as mu
import utils
import os
import datetime


def model_inputs(real_dim, z_dim):
    input_real = tf.placeholder(tf.float32, (None, *real_dim), name='input_real')
    input_z = tf.placeholder(tf.float32, (None, *z_dim), name='input_z')
    return input_real, input_z

def model_loss(input_real, image_dim, input_z, output_dim, alpha=0.2, label_smooth=1.0, drop_rate=0.,minibatch=False):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_dim: Image dimensions for the real input
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :param alpha: step size
	:param label_smooth: label smoothing factor
    :param minibatch: whether to use the minibatch or not
    :return: A tuple of (discriminator loss, generator loss)
    """
    g_model = mu.generator(input_z, image_dim, output_dim, alpha=alpha)

    d_model_real, d_logits_real, d_features_real = mu.discriminator(input_real, image_dim, alpha=alpha, drop_rate=drop_rate, minibatch=minibatch)
    d_model_fake, d_logits_fake, d_features_fake = mu.discriminator(g_model, image_dim, reuse=True, alpha=alpha, drop_rate=drop_rate, minibatch=minibatch)

    label_g = tf.ones_like(d_model_fake)
    label_d_real = tf.ones_like(d_model_real) - np.random.random_sample() * label_smooth
    label_d_fake = tf.zeros_like(d_model_fake) + np.random.random_sample() * label_smooth
    
    #g_loss = tf.reduce_mean(
    #    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=label_g))

    # Feature matching
    # This loss consists of minimizing the absolute difference between the expected features
    # on the data and the expected features on the generated samples.
    real_moments = tf.reduce_mean(d_features_real, axis=0)
    fake_moments = tf.reduce_mean(d_features_fake, axis=0)
    g_loss = tf.reduce_mean(tf.abs(real_moments - fake_moments))

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=label_d_real))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=label_d_fake))
    
    d_loss = d_loss_real + d_loss_fake
    
    return d_loss, g_loss

def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    # Get weights and bias to update
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]
    
    # Optimize, Using Adam optimizer
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt

class GAN:
    def __init__(self, real_size, z_size, image_size, learning_rate=0.0002, alpha=0.2, beta1=0.5, label_smooth=1.0, minibatch=False):
        tf.reset_default_graph()

        # Image size
        self.image_size = image_size

        # Drop rate
        self.drop_rate = tf.placeholder_with_default(.5, (), "drop_rate")
        
        # Create input place holders
        self.input_real, self.input_z = model_inputs(real_size, z_size)
        
        # Get the model losses
        self.d_loss, self.g_loss = model_loss(self.input_real, self.image_size, 
        										self.input_z, real_size[2], alpha=alpha, 
                                                label_smooth=label_smooth, drop_rate=self.drop_rate,
                                                minibatch=minibatch)
        
        # Get the optimized parameters
        self.d_opt, self.g_opt = model_opt(self.d_loss, self.g_loss, learning_rate, beta1=beta1)

def gen(model, z_size, use_gaussain=True, checkpoints_path='checkpoints'):
    saver = tf.train.Saver() # Saver used to save the checkpoints

    if use_gaussain:
        sample_z = np.random.normal(0, 1, size=(64, *z_size))
    else:
        sample_z = np.random.uniform(-1, 1, size=(64, *z_size)) 

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        # Try to restore the check point
        if os.listdir('./{}'.format(checkpoints_path)):
            saver.restore(sess, tf.train.latest_checkpoint('./{}'.format(checkpoints_path)))

        comic_gen = sess.run(mu.generator(model.input_z, model.image_size, 3, reuse=True, training=False),
                                    feed_dict={model.input_z: sample_z})

        # Save generated samples
        utils.save_samples(comic_gen, './output/generated')

def train(model, z_size, dataset, epochs, batch_size, print_every=10, show_every=100, use_gaussain=True, checkpoints_path='checkpoints'):
    saver = tf.train.Saver() # Saver used to save the checkpoints
    samples, losses = [], [] # Outputs
    
    if use_gaussain:
        sample_z = np.random.normal(0, 1, size=(64, *z_size))
    else:
        sample_z = np.random.uniform(-1, 1, size=(64, *z_size)) 
    
    steps = 0 # This variable is for showing the generator images
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Initialize the variables, this is a 
        # standard tensorflow operation
        sess.run(tf.global_variables_initializer())

        # Try to restore the check point
        if os.listdir('./{}'.format(checkpoints_path)):
            saver.restore(sess, tf.train.latest_checkpoint('./{}'.format(checkpoints_path)))
            print("Privious check point loaded...")
        
        # Loop through epochs...
        for e in range(epochs):
            for x in dataset.get_batches(batch_size):
                steps += 1
                
                # Sample random noise for G
                if use_gaussain:
                    batch_z = np.random.normal(0, 1, size=(batch_size, *z_size))
                else:
                    batch_z = np.random.uniform(-1, 1, size=(batch_size, *z_size))

                # Run optimizers
                _ = sess.run(model.d_opt, feed_dict={model.input_real: x, model.input_z: batch_z})
                _ = sess.run(model.g_opt, feed_dict={model.input_z: batch_z, model.input_real: x})
                
                # Display options
                if steps % print_every == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d = model.d_loss.eval({model.input_z:batch_z, model.input_real:x})
                    train_loss_g = model.g_loss.eval({model.input_z:batch_z, model.input_real:x})
                    now_time = datetime.datetime.now()

                    print(now_time.strftime("%Y-%m-%d %H:%M:%S"),
                          "Epoch {}/{}...".format(e+1, epochs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))
                    
                    # Save losses for later view
                    losses.append((train_loss_d, train_loss_g))
                    
                if steps % show_every == 0:
                    comic_gen = sess.run(
                                    mu.generator(model.input_z, model.image_size, 3, reuse=True, training=False),
                                    feed_dict={model.input_z: sample_z})
                    # Save generated samples
                    utils.save_samples(comic_gen, './output/gen_epoch_{}'.format(e+1))
                    
            saver.save(sess, './{}/cgan_epoch_{}'.format(checkpoints_path, e+1))  
   
    with open('samples.pkl', 'wb') as f:
        pkl.dump(samples, f)
        
    return losses, samples
