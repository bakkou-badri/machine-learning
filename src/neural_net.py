import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class NeuralNet:
    """ NeuralNet class
    An implementation of a multilayered artificial neural network
    for multiclass data classification. The dimension of first (input)
    layer MUST match the dimension of the data sample. The dimension
    of the last layer must match the number of classes (labels).
    """

    def __init__(self, learning_rate, network_shape):
        self.learning_rate = learning_rate
        self.network_shape = network_shape
        self.weights = self.init_weights(network_shape)
        self.biases = self.init_biases(network_shape)

        # Placeholders for inputs and labels
        self.x = tf.placeholder(tf.float32, [None, network_shape[0]])
        self.y_ = tf.placeholder(tf.float32, [None, network_shape[-1]])

        # Neural network model
        self.y = self.feed_forward(self.x, self.weights, self.biases)

        # Define loss J and optimizer
        self.J = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.J)

        # Define accuracy calculation
        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def train(self, max_iter, data_set):
        """ Trains the network
        """
        # Start tensorflow session and init variables
        sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()

        for i in range(max_iter):
            batch = data_set.train.next_batch(50)
            self.optimizer.run({self.x: batch[0], self.y_: batch[1]})
            if i % 10 == 0:
                print("Iteration: %s, Accuracy (ts): %s, Accuracy (vs): %s" %
                      (i, self.accuracy.eval({self.x: batch[0],
                                              self.y_: batch[1]}),
                       self.accuracy.eval({self.x: data_set.test.images,
                                           self.y_: data_set.test.labels})))

    def init_weights(self, shape):
        """ Randomly initializes weight matrices.
        :param shape: network shape tuple
        :return: weight matrices
        """
        weights = {}
        for i in range(len(shape)-1):
            weights[i] = tf.Variable(tf.random_normal([shape[i], shape[i+1]]))
        return weights

    def init_biases(self, shape):
        """ Randomly initializes biases vectors.
        :param shape: network shape tuple
        :return: bias vectors
        """
        biases = {}
        for i in range(len(shape)-1):
            biases[i] = tf.Variable(tf.random_normal([shape[i+1]]))
        return biases

    def feed_forward(self, x, weights, biases):
        """ Creates the neural network representation and workflow.
        :param x: input matrix
        :param weights: weight matrices
        :param biases: bias vectors
        :return: UNSCALED output of the last layer
        """
        l = None
        for i in range(len(weights)):
            if l is None:
                # Use X as an input for the first layer.
                l = tf.nn.relu(tf.add(tf.matmul(x, weights[i]), biases[i]))
            elif i < len(weights) - 1:
                # Use the output of the previous layer as the inputs for all
                # hidden layers.
                l = tf.nn.relu(tf.add(tf.matmul(l, weights[i]), biases[i]))
            else:
                # Don't scale the output of the last layer as it's taken
                # care of within the softmax_cross_entropy_with_logits
                # function when calculating the error.
                l = (tf.add(tf.matmul(l, weights[i]), biases[i]))
        return l

def main():
    learning_rate = 0.01

    # Define network shape as follows:
    #  * input layer matches the data dimension (784 pixels per image).
    #  * two hidden layer contains 256 neurons each.
    #  * last layer contains 10 neurons which is also the number of classes.
    # Note: while the dimension of the first and the last layer must match
    # dataset parameters, the number of hidden layers and the number of neurons
    # in these layers can be customised and will affect the classification
    network_shape = (784, 256, 256, 10)
    
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    nn = NeuralNet(learning_rate, network_shape)
    nn.train(20000, mnist)

if __name__ == "__main__":
    main()
