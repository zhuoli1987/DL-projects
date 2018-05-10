import numpy as np
import tensorflow as tf

def generator(z, image_size, output_dim, reuse=False, alpha=0.2, training=True):
    with tf.variable_scope('generator', reuse=reuse):
        
        # First fully connected layer
        #layer1 = tf.layers.dense(z, image_dim*image_dim*1024, name='g_dense_0')
        #layer1 = tf.reshape(layer1, (-1, image_dim, image_dim, 1024))
        #layer1 = tf.layers.batch_normalization(layer1, training=training)
        #layer1 = tf.nn.leaky_relu(layer1, alpha=alpha)

        # First conv layer
        layer2 = tf.layers.conv2d_transpose(z, image_size*8, 4, strides=1, padding='valid')
        layer2 = tf.layers.batch_normalization(layer2, training=training)
        layer2 = tf.nn.leaky_relu(layer2, alpha=alpha)
        
        # Second conv layer
        layer3 = tf.layers.conv2d_transpose(layer2, image_size*4, 4, strides=2, padding='same')
        layer3 = tf.layers.batch_normalization(layer3, training=training)
        layer3 = tf.nn.leaky_relu(layer3, alpha=alpha)
        
        # Third conv layer
        layer4 = tf.layers.conv2d_transpose(layer3, image_size*2, 4, strides=2, padding='same')
        layer4 = tf.layers.batch_normalization(layer4, training=training)
        layer4 = tf.nn.leaky_relu(layer4, alpha=alpha)

        # Forth conv layer
        layer5 = tf.layers.conv2d_transpose(layer4, image_size, 4, strides=2, padding='same')
        layer5 = tf.layers.batch_normalization(layer5, training=training)
        layer5 = tf.nn.leaky_relu(layer5, alpha=alpha)

        # Fifth conv layer
        layer6 = tf.layers.conv2d(layer5, image_size, 3, strides=1, padding='same')
        layer6 = tf.layers.batch_normalization(layer6, training=training)
        layer6 = tf.nn.leaky_relu(layer6, alpha=alpha)

        # Output layer, (image_dim * layer^2)x(image_dim * layer^2)x3
        logits = tf.layers.conv2d_transpose(layer6, output_dim, 4, strides=2, padding='same')
        out = tf.tanh(logits)
        
        return out

def discriminator(x, image_size, reuse=False, alpha=0.2, drop_rate=0., minibatch=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        x = tf.layers.dropout(x, rate=drop_rate/2.5)

        # Input is (image_size)x(image_size))x3
        layer1 = tf.layers.conv2d(x, image_size, 4, strides=2, padding='same')
        relu1 = tf.nn.leaky_relu(layer1, alpha=alpha)
        relu1 = tf.layers.dropout(relu1, rate=drop_rate)

        # (image_size)x(image_size))x3
        layer2 = tf.layers.conv2d(relu1, image_size*2, 4, strides=2, padding='same')
        bn2 = tf.layers.batch_normalization(layer2, training=True)
        relu2 = tf.nn.leaky_relu(bn2, alpha=alpha)

        # (image_size/2)x(image_size/2)ximage_size*2
        layer3 = tf.layers.conv2d(relu2, image_size*4, 4, strides=2, padding='same')
        bn3 = tf.layers.batch_normalization(layer3, training=True)
        relu3 = tf.nn.leaky_relu(bn3, alpha=alpha)
        relu3 = tf.layers.dropout(relu3, rate=drop_rate)

        # (image_size/4)x(image_size/4)ximage_size*4
        layer4 = tf.layers.conv2d(relu3, image_size*8, 4, strides=2, padding='same')
        bn4 = tf.layers.batch_normalization(layer4, training=True)
        relu4 = tf.nn.leaky_relu(bn4, alpha=alpha)

        # (image_size/4)x(image_size/4)x1024
        layer5 = tf.layers.conv2d(relu4, 1024, 4, strides=1, padding='valid')
        relu5 = tf.nn.leaky_relu(layer5, alpha=alpha)

        # 1024 x 1 x 1
        #flat = tf.reshape(relu5, (-1, 1024))
        flat = tf.reduce_mean(relu5, (1,2))

        if minibatch:
            features = mini_batch(flat)
        else:
            features = flat

        logits = tf.layers.dense(features, 1)
        out = tf.sigmoid(logits)

        return out, logits, features

def linear(inputs, output_dim, scope=None, stddev=1.0):
    """
    Matrix multiplication utility
    """
    with tf.variable_scope(scope or 'linear'):
        W = tf.get_variable("W",
                shape=[inputs.get_shape()[1], output_dim],
                initializer=tf.random_normal_initializer(stddev=stddev))

        b = tf.get_variable("b",
                shape=[output_dim],
                initializer=tf.constant_initializer(0.0))

    return tf.matmul(inputs, W) + b    

def mini_batch(inputs, kernels=5, kernel_dim=3):
    """
    Used to solve the collapse of the generator problem
    Steps
    :Multiply the input (f(x_i)) by a tensor (T) (Matrix M_i)
    :Calculate the absolute difference of L1-distance between 
    : this sample and all other samples in the batch for each row
    :Apply negative exponential (c_b(x_i, x_j) = exp(-||M_ib - M_jb||_L1))
    :Sum all the c_b (o(x_i) = sum(c_b(x_i, x_j)))
    :Concate the sum values to the output [o(x_i)1, o(x_i)2, ...]
    """
    m_i = linear(inputs, kernels * kernel_dim, scope='minibatch', stddev=0.02)
    m_i = tf.reshape(m_i, (-1, kernels, kernel_dim))

    diff_l1 = tf.expand_dims(m_i, 3) - tf.expand_dims(tf.transpose(m_i, [1, 2, 0]), 0)
    #eps = tf.expand_dims(np.eye(int(inputs.get_shape()[0]), dtype=np.float32), 1)
    abs_diffs = tf.reduce_sum(tf.abs(diff_l1), 2) 

    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([inputs, minibatch_features], 1)