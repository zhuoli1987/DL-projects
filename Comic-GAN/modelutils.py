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
        layer2 = tf.layers.conv2d_transpose(z, image_size*8, 4, strides=1, padding='valid', name='g_convt_1')
        layer2 = tf.layers.batch_normalization(layer2, training=training)
        layer2 = tf.nn.leaky_relu(layer2, alpha=alpha)
        
        # Second conv layer
        layer3 = tf.layers.conv2d_transpose(layer2, image_size*4, 4, strides=2, padding='same', name='g_convt_2')
        layer3 = tf.layers.batch_normalization(layer3, training=training)
        layer3 = tf.nn.leaky_relu(layer3, alpha=alpha)
        
        # Third conv layer
        layer4 = tf.layers.conv2d_transpose(layer3, image_size*2, 4, strides=2, padding='same',name='g_convt_3')
        layer4 = tf.layers.batch_normalization(layer4, training=training)
        layer4 = tf.nn.leaky_relu(layer4, alpha=alpha)

        # Forth conv layer
        layer5 = tf.layers.conv2d_transpose(layer4, image_size, 4, strides=2, padding='same',name='g_convt_4')
        layer5 = tf.layers.batch_normalization(layer5, training=training)
        layer5 = tf.nn.leaky_relu(layer5, alpha=alpha)

        # Fifth conv layer
        layer6 = tf.layers.conv2d(layer5, image_size, 3, strides=1, padding='same',name='g_conv_5')
        layer6 = tf.layers.batch_normalization(layer6, training=training)
        layer6 = tf.nn.leaky_relu(layer6, alpha=alpha)

        # Output layer, (image_dim * layer^2)x(image_dim * layer^2)x3
        logits = tf.layers.conv2d_transpose(layer6, output_dim, 4, strides=2, padding='same',name='g_convt_6')
        out = tf.tanh(logits)
        
        return out

def discriminator(x, image_size, reuse=False, alpha=0.2):
    with tf.variable_scope('discriminator', reuse=reuse):
        
        # Input is (image_size)x(image_size))x3
        layer1 = tf.layers.conv2d(x, image_size, 4, strides=2, padding='same', name='d_conv_1')
        relu1 = tf.nn.leaky_relu(layer1, alpha=alpha)

        # (image_size)x(image_size))x3
        layer2 = tf.layers.conv2d(relu1, image_size*2, 4, strides=2, padding='same', name='d_conv_2')
        bn2 = tf.layers.batch_normalization(layer2, training=True)
        relu2 = tf.nn.leaky_relu(bn2, alpha=alpha)

        # (image_size/2)x(image_size/2)ximage_size*2
        layer3 = tf.layers.conv2d(relu2, image_size*4, 4, strides=2, padding='same', name='d_conv_3')
        bn3 = tf.layers.batch_normalization(layer3, training=True)
        relu3 = tf.nn.leaky_relu(bn3, alpha=alpha)

        # (image_size/4)x(image_size/4)ximage_size*4
        layer4 = tf.layers.conv2d(relu3, image_size*8, 4, strides=2, padding='same', name='d_conv_4')
        bn4 = tf.layers.batch_normalization(layer4, training=True)
        relu4 = tf.nn.leaky_relu(bn4, alpha=alpha)
        drop4 = tf.layers.dropout(relu4, rate=0.5)

        # 1024 x 1 x 1
        flat = tf.reshape(drop4, (-1, 1024))
        logits = tf.layers.dense(flat, 1)
        out = tf.sigmoid(logits)

        return out, logits