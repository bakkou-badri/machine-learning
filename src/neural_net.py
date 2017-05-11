import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class NeuralNet:
    """ NeuralNet class
    An implementation of a multilayered artificial neural network
    for multiclass data classification. The dimension of first (input)
    layer MUST match the dimension of the data sample. The dimension
    of the last layer must match the number of classes (labels).
    """

    def __init__(self, learning_rate, shape, session):
        self.session = session
        self.learning_rate = learning_rate
        self.shape = shape
        self.weights = self.init_weights()
        self.biases = self.init_biases()

        # Placeholders for inputs and labels
        self.x = tf.placeholder(tf.float32, [None, shape[0]])
        self.y_ = tf.placeholder(tf.float32, [None, shape[-1]])

        # Neural network model
        self.y = self.feed_forward()

        # Define loss J and optimizer
        self.J = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
            logits=self.y, labels=self.y_))
        self.J_summary_train = tf.summary.scalar('cross_entropy_train', self.J)
        self.J_summary_valid = tf.summary.scalar('cross_entropy_valid', self.J)
        """
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.J)
        """
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate).minimize(self.J)
        """
        self.optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=0.9).minimize(self.J)
        """
        """
        self.optimizer = tf.train.AdagradOptimizer(
            learning_rate=learning_rate).minimize(self.J)
        """
        # Define accuracy calculation
        self.correct_prediction = tf.equal(
            tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32))
        self.train_summary = tf.summary.scalar('accuracy_train', self.accuracy)
        self.validation_summary = tf.summary.scalar('accuracy_validation', self.accuracy)
        #self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("./logs/mlp/", session.graph)
        tf.global_variables_initializer().run(session=self.session)

    def train(self, max_iter, data_set):
        """ Trains the network
        """

        for i in range(max_iter):
            batch = data_set.train.next_batch(50)
            _, t_summary, j = self.session.run([self.optimizer, self.train_summary, self.J_summary_train], {self.x: batch[0], self.y_: batch[1]})
            self.writer.add_summary(t_summary, i)
            self.writer.add_summary(j, i)
            v_summary, j_valid = self.session.run([self.validation_summary, self.J_summary_valid], {self.x: data_set.test.images, self.y_: data_set.test.labels})
            self.writer.add_summary(v_summary, i)
            self.writer.add_summary(j_valid, i)
            """
            self.optimizer.run({self.x: batch[0], self.y_: batch[1]},
                               session=self.session)
            """
            """
            if i % 10 == 0:
                print("Iteration: %s, Accuracy (ts): %s, Accuracy (vs): %s" %
                      (i, self.accuracy.eval({self.x: batch[0],
                                              self.y_: batch[1]},
                                             session=self.session),
                       self.accuracy.eval({self.x: data_set.test.images,
                                           self.y_: data_set.test.labels},
                                          session=self.session)))
            """

    def init_weights(self):
        """ Randomly initializes weight matrices.
        :return: weight matrices
        """
        weights = {}
        for i in range(len(self.shape) - 1):
            weights[i] = tf.Variable(tf.random_normal(
                [self.shape[i], self.shape[i + 1]]))
        return weights

    def init_biases(self):
        """ Randomly initializes biases vectors.
        :return: bias vectors
        """
        biases = {}
        for i in range(len(self.shape) - 1):
            biases[i] = tf.Variable(tf.random_normal([self.shape[i + 1]]))
        return biases

    def feed_forward(self):
        """ Creates the neural network representation and workflow.
        :return: UNSCALED output of the last layer
        """
        l = None
        layer_idx = len(self.shape) - 1
        for i in range(layer_idx):
            if l is None:
                # Use X as an input for the first layer.
                l = tf.nn.relu(
                    tf.add(tf.matmul(self.x, self.weights[i]), self.biases[i]))
            elif i < layer_idx - 1:
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
    learning_rate = 0.005
    max_iter = 10000

    # Define network shape as follows:
    #  * input layer matches the data dimension (784 pixels per image).
    #  * two hidden layer contains 256 neurons each.
    #  * last layer contains 10 neurons which is also the number of classes.
    # Note: while the dimension of the first and the last layer must match
    # dataset parameters, the number of hidden layers and the number of neurons
    # in these layers can be customised and will affect the classification
    shape = (784, 1024, 1024, 10)

    mnist = input_data.read_data_sets("../data/mnist/", one_hot=True)

    session = tf.Session(config=tf.ConfigProto(log_device_placement=True, device_count = {'GPU': 0}))
    nn = NeuralNet(learning_rate, shape, session)
    nn.train(max_iter, mnist)
    session.close()

if __name__ == "__main__":
    main()
