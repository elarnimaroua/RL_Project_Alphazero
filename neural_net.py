
import os

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


class NeuralNetwork(object):
    """Represents the Policy and Value Resnet.
    """

    def __init__(self, game):
        """Initializes NeuralNetwork with the Resnet network graph."""
        self.row = game.row
        self.column = game.column
        self.action_size = self.row * self.column
        self.pi = None #tensor for the search probabilities
        self.v = None #tensor for the search values
        self.resnet_blocks = 20
        self.learning_rate = 0.001
        self.epochs = 10
        self.batch_size = 64
        self.model_directory = "./connect_four/models/"
        self.record_loss = 1
        self.loss_file = "loss.txt"
        self.graph = tf.Graph()
        self.momentum = 0.9

        with self.graph.as_default():
            self.states = tf.placeholder(tf.float32, shape=[None, self.row, self.column]) #tensor with the dimensions of the board.
            self.training = tf.placeholder(tf.bool) 
            input_layer = tf.reshape(self.states, [-1, self.row, self.column, 1])
            conv1 = tf.layers.conv2d(inputs=input_layer, filters=256, kernel_size=[3, 3], padding="same", strides=1)
            batch_norm1 = tf.layers.batch_normalization(inputs=conv1, training=self.training)
            relu1 = tf.nn.relu(batch_norm1)
            resnet_in_out = relu1

            for i in range(self.resnet_blocks):
                # Residual Block
                conv2 = tf.layers.conv2d(inputs=resnet_in_out, filters=256, kernel_size=[3, 3], padding="same", strides=1)
                batch_norm2 = tf.layers.batch_normalization(inputs=conv2, training=self.training)
                relu2 = tf.nn.relu(batch_norm2)
                conv3 = tf.layers.conv2d(inputs=relu2, filters=256, kernel_size=[3, 3], padding="same", strides=1)
                batch_norm3 = tf.layers.batch_normalization(inputs=conv3, training=self.training)
                resnet_skip = tf.add(batch_norm3, resnet_in_out)
                resnet_in_out = tf.nn.relu(resnet_skip)

            #Policy head
            conv4 = tf.layers.conv2d(inputs=resnet_in_out, filters=2, kernel_size=[1, 1], padding="same", strides=1)
            batch_norm4 = tf.layers.batch_normalization(inputs=conv4, training=self.training)
            relu4 = tf.nn.relu(batch_norm4)
            relu4_flat = tf.reshape(relu4, [-1, self.row * self.column * 2])
            logits = tf.layers.dense(inputs=relu4_flat, units=self.action_size)
            self.pi = tf.nn.softmax(logits)
            
            #value head
            conv5 = tf.layers.conv2d(inputs=resnet_in_out, filters=1, kernel_size=[1,1], padding="same", strides=1)
            batch_norm5 = tf.layers.batch_normalization(inputs=conv5,training=self.training)
            relu5 = tf.nn.relu(batch_norm5)
            relu5_flat = tf.reshape(relu5, [-1, self.action_size])
            dense1 = tf.layers.dense(inputs=relu5_flat, units=256)
            relu6 = tf.nn.relu(dense1)
            dense2 = tf.layers.dense(inputs=relu6,units=1)
            self.v = tf.nn.tanh(dense2)

            #Calculating loss
            self.train_pis = tf.placeholder(tf.float32,shape=[None, self.action_size]) #tensor for the target search probabilities
            self.train_vs = tf.placeholder(tf.float32, shape=[None]) #tensor for the target search values
            self.loss_pi = tf.losses.softmax_cross_entropy(self.train_pis, self.pi) #tensor for the softmax crossentropy on pi
            self.loss_v = tf.losses.mean_squared_error(self.train_vs, tf.reshape(self.v, shape=[-1, ])) #tensor for the mean squared error on v
            self.total_loss = self.loss_pi + self.loss_v #tensor to store the addition of pi and v 
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum, use_nesterov=False)
            self.train_op = optimizer.minimize(self.total_loss) #tensor for output of the optimizer
            self.saver = tf.train.Saver() #tf saver for writing training checkpoints
            self.sess = tf.Session() #tf session for running Ops on the Graph
            self.sess.run(tf.global_variables_initializer())  #Initialize the session


