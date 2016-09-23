import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class LogReg:
    """ LogReg class
    An implemetation of logistic regression for multiclass data classification
    """

    def __init__(self, learning_rate, shape):
        if len(shape) != 2:
            print "Length of the data shape must be 2! (NO_FEATURES, NO_CLASSES)"
            print "Exiting!"
            exit(1)

        self.shape = shape
        self.display_step = 100
        self.learning_rate = learning_rate
        self.W = tf.Variable(tf.zeros([shape[0], shape[1]]))
        self.b = tf.Variable(tf.zeros([shape[1]]))

        # Placeholders for inputs and labels
        self.x = tf.placeholder(tf.float32, [None, shape[0]])
        self.y_ = tf.placeholder(tf.float32, [None, shape[1]])

        # Logistic regression model
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)

        # Define loss J and optimizer
        self.J = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y),
                                               reduction_indices=[1]))
        #self.J = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.J)

        # Define accuracy calculation
        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def train(self, max_iter, data):
        """ Trains the model
        """
        # Start tensorflow session and init variables
        sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()

        for i in range(max_iter):
            batch = data.train.next_batch(50)
            self.optimizer.run({self.x: batch[0], self.y_: batch[1]})
            if i % self.display_step == 0:
                print("Iteration: %s, Accuracy (ts): %s, Accuracy (vs): %s" %
                      (i, self.accuracy.eval({self.x: batch[0],
                                              self.y_: batch[1]}),
                       self.accuracy.eval({self.x: data.test.images,
                                      self.y_: data.test.labels})))
def main():
    learning_rate = 0.01
    max_iter = 10000

    # Define input and output dimensions as follows:
    #  * (INPUT_DIMENSION, NUMBER_OF_CLASSES)
    # The len(shape) for logistic regression MUST be 2!
    shape = (784, 10)

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    lr = LogReg(learning_rate, shape)
    lr.train(max_iter, mnist)

if __name__ == "__main__":
    main()
