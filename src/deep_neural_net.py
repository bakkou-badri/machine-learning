import tensorflow as tf
import autoencoder as ae
from tensorflow.examples.tutorials.mnist import input_data


class DeepNeuralNet:
    """ DeepNeuralNet class
    An implementation of a multilayered artificial neural network
    for multiclass data classification. The dimension of first (input)
    layer MUST match the dimension of the data sample. The dimension
    of the last layer must match the number of classes (labels).
    This class (unlike the standard NeuralNet class) implements
    methods for pretraining weights and biases.
    """

    def __init__(self, learning_rate, shape, session):
        print("Creating (Deep) Neural Net")
        print("Shape: %s" % str(shape))
        self.session = session
        self.learning_rate = learning_rate
        self.shape = shape
        self.display_step = 100

        self.weights = self.init_weights()
        self.biases = self.init_biases()

        # Placeholders for inputs and labels
        self.x = tf.placeholder(tf.float32, [None, shape[0]])
        self.y_ = tf.placeholder(tf.float32, [None, shape[-1]])

        # Neural network model
        self.y = self.feed_forward()

        # Define loss J and optimizer
        self.J = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.J)

        # Define accuracy calculation
        self.correct_prediction = tf.equal(
            tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32))
        tf.initialize_all_variables().run(session=self.session)

    def pretrain_layer(self, layer_idx, max_iter, dataset):
        """ Pretrains network layer using autoencoder.
        """
        shape = (self.shape[layer_idx],
                 self.shape[layer_idx + 1],
                 self.shape[layer_idx])
        lae = ae.Autoencoder(0.01, shape, self.session)
        for i in range(max_iter):
            x, y = dataset.train.next_batch(50)
            if layer_idx > 0:
                # Get representation of the input on the layer_idx'th layer
                x = self.feed_forward(layer_idx).eval({self.x: x},
                                                      session=self.session)
            lae.train_batch((x,))
        self.weights[layer_idx] = tf.Variable(lae.weights[0])
        self.biases[layer_idx] = tf.Variable(lae.biases[0])

    def pretrain(self, max_iter, dataset):
        """ Pretrains all networks' layers
        This pretrains weights and biases so the information
        can spread within the network with as little noise
        as possible.
        :param: max_iter: number of iteration for pretraining
        :dataset: training dataset
        """
        for i in range(len(self.shape) - 1):
            self.pretrain_layer(i, max_iter, dataset)

    def init_weights(self):
        """ Init weights randomly.
        """
        weights = {}
        for i in range(len(self.shape) - 1):
            weights[i] = tf.Variable(tf.random_normal(
                [self.shape[i], self.shape[i + 1]]))
        return weights

    def init_biases(self):
        """ Init biases randomly.
        """
        biases = {}
        for i in range(len(self.shape) - 1):
            biases[i] = tf.Variable(tf.random_normal([self.shape[i + 1]]))
        return biases

    def train(self, max_iter, dataset):
        """ Trains the network
        """
        for i in range(max_iter):
            batch = dataset.train.next_batch(50)
            self.optimizer.run({self.x: batch[0], self.y_: batch[1]},
                               session=self.session)
            if i % self.display_step == 0:
                print("Iteration: %s, Accuracy (ts): %s, Accuracy (vs): %s" %
                      (i, self.accuracy.eval({self.x: batch[0],
                                              self.y_: batch[1]},
                                             session=self.session),
                       self.accuracy.eval({self.x: dataset.test.images,
                                           self.y_: dataset.test.labels},
                                          session=self.session)))

    def feed_forward(self, layer_idx=-1):
        """ Creates the neural network representation and workflow.
        :return: UNSCALED output of the last layer
        """
        l = None
        if layer_idx < 1:
            layer_idx = len(self.shape) - 1
        for i in range(layer_idx):
            if l is None:
                # Use X as an input for the first layer.
                l = tf.nn.relu(
                    tf.add(tf.matmul(self.x, self.weights[i]), self.biases[i]))
            elif i < len(self.weights) - 1:
                # Use the output of the previous layer as the inputs for all
                # hidden layers.
                l = tf.nn.relu(
                    tf.add(tf.matmul(l, self.weights[i]), self.biases[i]))
            else:
                # Don't scale the output of the last layer as it's taken
                # care of within the softmax_cross_entropy_with_logits
                # function when calculating the error.
                l = (tf.add(tf.matmul(l, self.weights[i]), self.biases[i]))
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
    shape = (784, 256, 256, 10)

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    session = tf.Session()
    dnn = DeepNeuralNet(learning_rate, shape, session)
    dnn.pretrain(1000, mnist)
    dnn.train(10000, mnist)
    session.close()

if __name__ == "__main__":
    main()
