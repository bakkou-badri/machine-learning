import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class Autoencoder:
    """ Autoencoder class
    Symetric 3-layer neural network reconstructing X' from X.
    """

    def __init__(self, learning_rate, shape, session):
        if len(shape) != 3 or shape[0] != shape[-1]:
            print "Invalid shape %s" % (shape)
            print "Autoencoder must be a symetric 3-layered network"
            print "Exiting!"
            exit(-1)
        print "Creating autoencoder"
        print "Shape: %s" % str(shape)
        self.session = session
        self.display_step = 100
        self.learning_rate = learning_rate
        self.shape = shape
        self.weights = self.init_weights()
        self.biases = self.init_biases()

        # Placeholders for inputs and labels
        self.x = tf.placeholder(tf.float32, [None, shape[0]])
        self.y_ = tf.placeholder(tf.float32, [None, shape[-1]])

        # Autoencoder model
        self.y = self.feed_forward()

        # Define loss J and optimizer
        self.J = tf.reduce_mean(tf.pow(self.y - self.y_, 2))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.J)
        tf.initialize_all_variables().run(session=self.session)

    def train(self, max_iter, data):
        """ Trains the autoencoder
        """
        for i in range(max_iter):
            batch = data.train.next_batch(50)
            self.train_batch(batch)
            if i % self.display_step == 0:
                print "Iteration: %s, J: %s" % (i, self.J.eval({self.x: batch[0],
                                                                self.y_: batch[0]},
                                                session=self.session))

    def train_batch(self, batch):
        """ Trains one iteration using one batch of data.
        :param batch: a tuple containing (x, y)
        """
        self.optimizer.run({self.x: batch[0], self.y_: batch[0]},
                           session=self.session)

    def init_weights(self):
        weights = {}
        for i in range(len(self.shape) - 1):
            weights[i] = tf.Variable(tf.random_normal(
                [self.shape[i], self.shape[i + 1]]))
        return weights

    def init_biases(self):
        biases = {}
        for i in range(len(self.shape) - 1):
            biases[i] = tf.Variable(tf.random_normal([self.shape[i + 1]]))
        return biases

    def feed_forward(self, layer_idx=-1):
        """ Feeds the network forward up to layer_idx.
        The network is fed all the way through if the layer_idx
        parameter is not specified. If the layer_idx is specified,
        the network is fed UP TO layer_idx value. Therefore the
        layer_idx value must be within the <1, len(self.shape) -1>
        interval.
        :param layer_idx: layer index
        :return: output of layer[layer_idx]
        """
        l = None
        if layer_idx < 0:
            # If layer_idx not specified explicitly, feed all layers
            layer_idx = len(self.shape) - 1
        for i in range(layer_idx):
            if l is None:
                l = tf.nn.sigmoid(
                    tf.add(tf.matmul(self.x, self.weights[i]), self.biases[i]))
            else:
                # Don't use activation function on the output of the last layer
                l = tf.add(tf.matmul(l, self.weights[i]), self.biases[i])
        return l


def main():
    learning_rate = 0.01
    max_iter = 10000

    shape = (784, 128, 784)

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    session = tf.Session()
    ae = Autoencoder(learning_rate, shape, session)
    ae.train(max_iter, mnist)
    session.close()

if __name__ == "__main__":
    main()
