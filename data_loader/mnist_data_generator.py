from tensorflow import keras
import tensorflow as tf
import numpy as np


class MnistData:
    def __init__(self, config):
        self.config = config
        (x_train_, self.y_train), (x_test_, self.y_test) = keras.datasets.mnist.load_data()
        x_train_ = x_train_ / 255.
        x_test_ = x_test_ / 255.
        self.x_train = x_train_.reshape((-1, 28, 28, 1))
        self.x_test = x_test_.reshape((-1, 28, 28, 1))

        self.next_batch(self.config.batch_size)

    def next_batch(self, batch_size):
        # train_data
        train_data = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(self.x_train, dtype=tf.float32),
                                                         tf.convert_to_tensor(self.y_train, dtype=tf.int64)))
        train_data = train_data.shuffle(60000)
        train_data = train_data.batch(batch_size)

        # test_data
        test_data = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(self.x_test, dtype=tf.float32),
                                                         tf.convert_to_tensor(self.y_test, dtype=tf.int64)))
        test_data = test_data.batch(batch_size)

        # iterator
        iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)

        self.next_data = iterator.get_next()

        # init_op
        self.train_init_op = iterator.make_initializer(train_data)
        self.test_init_op = iterator.make_initializer(test_data)



if __name__ == '__main__':
    mnist = MnistData('')
    print(mnist.x_train.shape)
    train_init_op, test_init_op = mnist.next_batch(1)

    with tf.Session() as sess:
        sess.run(train_init_op)
        train_data, train_label = mnist.next_data
        print(sess.run(train_data).shape)
        print(sess.run(train_label))

