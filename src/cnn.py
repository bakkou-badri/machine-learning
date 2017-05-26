import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class ConvNet:
    """ ConvNet class
    An implementation of a deep convolutional neural network for
    image classification.
    """

    def __init__(self, learning_rate, shape, session, img_dims):
        self.session = session
        self.learning_rate = learning_rate
        self.shape = shape
        self.weights = self.init_weights()
        self.biases = self.init_biases()
        self.display_step = 100
        self.img_dims = img_dims

        self.x = tf.placeholder(tf.float32, [None, img_dims[0] * img_dims[1]])
        self.y_ = tf.placeholder(tf.float32, [None, self.shape["out"][-1]])

        self.y = self.feed_forward()

        self.J = tf.reduce_mean(-tf.reduce_sum(self.y_ *
                                               tf.log(self.y), reduction_indices=[1]))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.J)
        self.correct_prediction = tf.equal(
            tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32))

        tf.global_variables_initializer().run(session=self.session)

    def train(self, max_iter, dataset):
        for i in range(max_iter):
            batch = dataset.train.next_batch(50)
            self.optimizer.run({self.x: batch[0], self.y_: batch[1]},
                               session=self.session)
            if i % self.display_step == 0:
                print("Iteration: %s, Accuracy (ts): %s, J: %s" %
                      (i, self.accuracy.eval({self.x: batch[0], self.y_: batch[1]},
                                             session=self.session),
                       self.J.eval({self.x: batch[0], self.y_: batch[1]},
                                   session=self.session)))

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def init_weight(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def init_bias(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def init_weights(self):
        weights = {}
        for k in self.shape.keys():
            weights[k] = self.init_weight(self.shape[k])
        return weights

    def init_biases(self):
        biases = {}
        for k in self.shape.keys():
            biases[k] = self.init_bias([self.shape[k][-1]])
        return biases

    def feed_forward(self):
        # Reshape image for convolution [-1, width, height, no_channels]
        x_image = tf.reshape(self.x, [-1, self.img_dims[0],
                                      self.img_dims[1], self.img_dims[2]])

        # First conv layer
        l_cv1 = tf.nn.relu(self.conv2d(
            x_image, self.weights["cv1"]) + self.biases["cv1"])
        p_cv1 = self.max_pool_2x2(l_cv1)

        # Second conv layer
        l_cv2 = tf.nn.relu(self.conv2d(
            p_cv1, self.weights["cv2"]) + self.biases["cv2"])
        p_cv2 = self.max_pool_2x2(l_cv2)

        # FC layer DO no pooling and reshaping
        h_pool2_flat = tf.reshape(p_cv2, [-1, self.shape["fc"][0]])
        h_fc1 = tf.nn.relu(
            tf.matmul(h_pool2_flat, self.weights["fc"]) + self.biases["fc"])

        # Readout layer
        y = tf.nn.softmax(tf.matmul(h_fc1, self.weights[
                          "out"]) + self.biases["out"])
        return y


def main():
    learning_rate = 0.0001

    # Define network shape
    # 2 convolutional layer, 1 fully connected layer and output layer
    shape = {"cv1": [5, 5, 1, 64],
             "cv2": [5, 5, 64, 128],
             "fc": [7 * 7 * 128, 1024],
             "out": [1024, 10]}

    mnist = input_data.read_data_sets("../data/mnist/", one_hot=True)

    # Define input image dimensions [width, height, channels]
    img_dims = [28, 28, 1]

    session = tf.Session()
    cnn = ConvNet(learning_rate, shape, session, img_dims)
    cnn.train(10000, mnist)
    session.close()

if __name__ == "__main__":
    main()
