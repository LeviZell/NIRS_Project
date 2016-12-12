import functools
import numpy as np
import pandas as pd

tf.reset_default_graph()
tf.Graph().as_default()

with open("tensorflow_csv_1.csv",'r') as f:
    with open("updated_tensorflow_csv_1.csv",'w') as f1:
        next(f) # skip header line
        for line in f:
            f1.write(line)
with open("tensorflow_csv_2.csv",'r') as f:
    with open("updated_tensorflow_csv_2.csv",'w') as f1:
        next(f) # skip header line
        for line in f:
            f1.write(line)

training_dataframe = pd.read_csv("tensorflow_csv_1.csv", skipinitialspace=True,
                            skiprows=1, names=COLUMNS)
test_dataframe = pd.read_csv("updated_tensorflow_csv_2.csv", skipinitialspace=True,
                            skiprows=1, names=COLUMNS)

COLUMNS = ["s680", "s720", "s760", "s800", "l680", "l720",
           "l760", "l800", "Normalized_O2_HHb", "Heart_Rate"]
FEATURES = ["s680", "s720", "s760", "s800", "l680", "l720",
           "l760", "l800", "Normalized_O2_HHb"]
feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]
# Data sets
OPTICAL_TRAINING = "updated_tensorflow_csv_1.csv"
OPTICAL_TEST = "updated_tensorflow_csv_2.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_without_header(OPTICAL_TRAINING, np.float, np.float)
test_set = tf.contrib.learn.datasets.base.load_csv_without_header(OPTICAL_TEST, np.float, np.float)

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class SequenceClassification:

    def __init__(self, data, target, dropout, num_hidden=21, num_layers=3):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        # Recurrent network.
        network = tf.nn.rnn_cell.GRUCell(self._num_hidden)
        network = tf.nn.rnn_cell.DropoutWrapper(
            network, output_keep_prob=self.dropout)
        network = tf.nn.rnn_cell.MultiRNNCell([network] * self._num_layers)
        output, _ = tf.nn.dynamic_rnn(network, data, dtype=tf.float32)
        # Select last output.
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)
        # Softmax layer.
        weight, bias = self._weight_and_bias(
            self._num_hidden, int(self.target.get_shape()[1]))
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = 0.003
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
#         mistakes = tf.logical_or((tf.less(tf.argmax(target, 1), (tf.argmax(prediction, 1)-1))), \
#             (tf.greater(tf.argmax(target, 1), (tf.argmax(prediction, 1)+1))))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


if __name__ == '__main__':
    # We treat images as sequences of pixel rows.
#     train, test = sets.Mnist()
    tf.Graph().as_default()
    train = training_set
    test = test_set
    print (train.data.shape)
    print (test.data.shape)
#     test.data = tf.reshape([None, rows, row_size])
    rows, row_size = train.data.shape
    
#     rows, row_size = test.data.shape
    print (train.target[0])
    print (rows)
    print (row_size)
#     num_classes = train.target.shape[0]
#     print (num_classes)
    data = tf.placeholder(tf.float32, [None, rows, row_size])
    print("data")
    print(data)
    target = tf.placeholder(tf.float32, [None, 1])
    print ("target")
    print(target)
    dropout = tf.placeholder(tf.float32)
    model = SequenceClassification(data, target, dropout)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    batch_size = 10
    no_of_batches = int(len(train)/batch_size)
    epoch = 10
    
    for i in range(epoch):
        ptr = 0
        for j in range(no_of_batches):
#             inp, out = train[ptr:ptr+batch_size], train[ptr:ptr+batch_size]
#             ptr+=batch_size
            batch = train.sample(batch_size)
            sess.run(model.optimize, {data: batch.data, target: batch.target, dropout: 0.5})
#             sess.run(minimize,{data: inp, target: out})
#         print ("Epoch - ",str(i))
        error = sess.run(model.error, {data: test.data, target: test.target, dropout: 1})
        print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * error))
#     incorrect = sess.run(error,{data: test_input, target: test_output})
# sess.close()