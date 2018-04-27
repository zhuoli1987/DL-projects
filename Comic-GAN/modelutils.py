import numpy as np
import tensorflow as tf


def generator(z, image_size, output_dim, reuse=False, alpha=0.2, training=True):
    with tf.variable_scope('generator', reuse=reuse):
        
        # First fully connected layer
        image_dim = int(image_size/16)
        layer1 = tf.layers.dense(z, image_dim*image_dim*1024, name='g_dense_0')
        layer1 = tf.reshape(layer1, (-1, image_dim, image_dim, 1024))
        layer1 = tf.layers.batch_normalization(layer1, training=training)
        layer1 = tf.maximum(alpha * layer1, layer1) # Leaky ReLU
        
        # First conv layer
        layer2 = tf.layers.conv2d_transpose(layer1, image_size*8, 5, strides=2, padding='same', name='g_convt_1')
        layer2 = tf.layers.batch_normalization(layer2, training=training)
        layer2 = tf.maximum(alpha * layer2, layer2) # Leaky ReLU
        
        # Second conv layer
        layer3 = tf.layers.conv2d_transpose(layer2, image_size*4, 5, strides=2, padding='same', name='g_convt_2')
        layer3 = tf.layers.batch_normalization(layer3, training=training)
        layer3 = tf.maximum(alpha * layer3, layer3) # Leaky ReLU
        
        # Third conv layer
        layer4 = tf.layers.conv2d_transpose(layer3, image_size*2, 5, strides=2, padding='same',name='g_convt_3')
        layer4 = tf.layers.batch_normalization(layer4, training=training)
        layer4 = tf.maximum(alpha * layer4, layer4) # Leaky ReLU
        
        # Output layer, (image_dim * layer^2)x(image_dim * layer^2)x3
        logits = tf.layers.conv2d_transpose(layer4, output_dim, 5, strides=2, padding='same',name='g_convt_4')
        
        out = tf.tanh(logits)
        
        return out

def discriminator(x, image_size, reuse=False, alpha=0.2):
    with tf.variable_scope('discriminator', reuse=reuse):
        
        # Input is (image_size)x(image_size))x3
        layer1 = tf.layers.conv2d(x, image_size*2, 5, strides=2, padding='same', name='d_conv_1')
        relu1 = tf.maximum(alpha * layer1, layer1)

        # (image_size/2)x(image_size/2)ximage_size*2
        layer2 = tf.layers.conv2d(relu1, image_size*4, 5, strides=2, padding='same', name='d_conv_2')
        bn2 = tf.layers.batch_normalization(layer2, training=True)
        relu2 = tf.maximum(alpha * bn2, bn2)

        # (image_size/4)x(image_size/4)ximage_size*4
        layer3 = tf.layers.conv2d(relu2, image_size*8, 5, strides=2, padding='same', name='d_conv_3')
        bn3 = tf.layers.batch_normalization(layer3, training=True)
        relu3 = tf.maximum(alpha * bn3, bn3)

        # (image_size/8)x(image_size/8)ximage_size*8
        layer4 = tf.layers.conv2d(relu3, image_size*16, 5, strides=2, padding='same', name='d_conv_4')
        bn4 = tf.layers.batch_normalization(layer4, training=True)
        relu4 = tf.maximum(alpha * bn4, bn4)

        # (image_size/16)x(image_size/16)ximage_size*16
        image_dim = int(image_size/16)
        flat = tf.reshape(relu4, (-1, image_dim*image_dim*image_size*16))
        logits = tf.layers.dense(flat, 1)
        out = tf.sigmoid(logits)

        return out, logits