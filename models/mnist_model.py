from base.base_model import BaseModel
import tensorflow as tf
from data_loader.mnist_data_generator import MnistData


class MnistModel(BaseModel):
    def __init__(self, config, data_loader=None):
        super(MnistModel, self).__init__(config)

        self.data_loader = data_loader
        self.is_training = True

        self.x, self.y = self.data_loader.next_data

        self.build_model()
        self.init_saver()

    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.
        conv_1 = tf.layers.conv2d(
            self.x,
            filters=32,
            kernel_size=5,
            padding='same',
            activation=tf.nn.relu
        )

        pool_1 = tf.layers.max_pooling2d(
            conv_1,
            pool_size=2,
            strides=2
        )

        conv_2 = tf.layers.conv2d(
            pool_1,
            filters=64,
            kernel_size=5,
            padding='same',
            activation=tf.nn.relu
        )

        pool_2 = tf.layers.max_pooling2d(
            conv_2,
            pool_size=2,
            strides=2
        )

        pool_2_flatten = tf.layers.flatten(pool_2)

        dense_1 = tf.layers.dense(pool_2_flatten, 1024, activation=tf.nn.relu)

        dense_2 = tf.layers.dense(dense_1, 512, activation=tf.nn.relu)

        logits = tf.layers.dense(dense_2, 10)

        with tf.name_scope('loss'):
            self.cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=self.y))
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.cross_entropy,
                                                                                                       global_step=self.global_step_tensor)
            preds = tf.nn.softmax(logits)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds, 1), self.y), tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)