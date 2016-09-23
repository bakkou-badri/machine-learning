import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Autoencoder:
    """ Autoencoder class
    Symetric neural network reconstructing X' from X.
    """
    def __init__(self, learning_rate, network_shape, session):
        if len(network_shape) != 3 or network_shape[0] != network_shape[-1]:
            print "Invalid shape %s" % (network_shape)
            print "Autoencoder must be a symetric 3-layered network"
            print "Exiting!"
            exit(-1)
        self.session = session
        self.display_step = 100
        self.learning_rate = learning_rate
        self.network_shape = network_shape
        self.weights = self.init_weights(network_shape)
        self.biases = self.init_biases(network_shape)

        # Placeholders for inputs and labels
        self.x = tf.placeholder(tf.float32, [None, network_shape[0]])
        self.y_ = tf.placeholder(tf.float32, [None, network_shape[-1]])

        # Autoencoder model
        self.y = self.feed_forward(self.x, self.weights, self.biases)

        # Define loss J and optimizer
        self.J = tf.reduce_mean(tf.pow(self.y - self.y_, 2))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.J)
        tf.initialize_all_variables().run(session=self.session)

    def train(self, max_iter, data):
        """ Trains the autoencoder
        """
        for i in range(max_iter):
            batch = data.train.next_batch(50)
            self.optimizer.run({self.x: batch[0], self.y_: batch[0]},
                               session=self.session)
            if i % self.display_step == 0:
                print "Iteration: %s, J: %s" % (i, self.J.eval({self.x: batch[0],
                                                                self.y_: batch[0]},
                                                session=self.session))

    def init_weights(self, shape):
        weights = {}
        for i in range(len(shape)-1):
            weights[i] = tf.Variable(tf.random_normal([shape[i], shape[i+1]]))
        return weights

    def init_biases(self, shape):
        biases = {}
        for i in range(len(shape)-1):
            biases[i] = tf.Variable(tf.random_normal([shape[i+1]]))
        return biases

    def feed_forward(self, x, weights, biases):
        l = None
        for i in range(len(weights)):
            if l is None:
                l = tf.nn.sigmoid(tf.add(tf.matmul(x, weights[i]), biases[i]))
            else:
                l = tf.nn.sigmoid(tf.add(tf.matmul(l, weights[i]), biases[i]))
        return l

def main():
    learning_rate = 0.01
    max_iter = 10000

    network_shape = (784, 128, 784)

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    session = tf.Session()
    ae = Autoencoder(learning_rate, network_shape, session)
    ae.train(max_iter, mnist)
    session.close()

if __name__ == "__main__":
    main()
