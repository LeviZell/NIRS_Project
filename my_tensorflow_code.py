import functools
import numpy as np
import pandas as pd

tf.reset_default_graph()
tf.Graph().as_default()

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
        self.minimize

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
        print (self.target, self.prediction)
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
    
    def minimize(self):
        optimizer = tf.train.AdamOptimizer()
        cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)





if __name__ == '__main__':
    # We treat images as sequences of pixel rows.
#     train, test = sets.Mnist()
    tf.Graph().as_default()
    train = training_set
    test = test_set

    print ("train shape")
    print (len(train), len(train[0]))
    print ("test shape")
    print (len(test), len(test[0]))
#     train = tf.placeholder(tf.float32, [None, rows, row_size])
    print ("train.data shape, test.data shape")
    print (train.data.shape, test.data.shape)

    print ("train.target shape, test.target shape")
    print (train.target.shape, test.target.shape)
    
    _, rows = train.data.shape
    row_size = 1
#     rows, row_size = test.data.shape
    print ("train.target[0] value / shape[0], shape")
    print (train.target[0], train.target.shape[0], train.target.shape)
    print ("_, rows, row_size")
    print (_, rows, row_size)
#     num_classes = train.target.shape[0]
    num_classes = 1
    print ("num_classes")
    print (num_classes)
    data = tf.placeholder(tf.float32, [None, rows, row_size])
    print("data")
    print(data)
    target = tf.placeholder(tf.float32, [None, num_classes])
    print ("target")
    print(target)
    print ("new train")
    
    print ("test_target_new") 
#     test_target_new = test.target.reshape(len(test.target), 1)
    test_target_new = np.expand_dims(test.target, axis=1)
    print (test_target_new.shape)
#     test_target_new = tf.expand_dims(test.target, 1)

    print ("test_data_new shape")
#     test_data_new = test.data.reshape(test.data.shape[0], test.data.shape[1], 1)
    test_data_new = np.expand_dims(test.data, axis=2)
    print (test_data_new.shape)
#     test_data_new = tf.expand_dims(test.data, 2)
#     print (test_data_new.get_shape())
    
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
            inp, out = train[ptr:ptr+batch_size], train[ptr:ptr+batch_size]
            ptr+=batch_size
            batch = train.sample(batch_size)
#             sess.run(minimize,{data: inp, target: out})

            sess.run(model.optimize, {data: batch.data, target: batch.target, dropout: 0.5})
#         print "Epoch ",str(i)
#         print (batch)

#             sess.run(minimize,{data: inp, target: out})
#         print ("Epoch - ",str(i))
        error = sess.run(model.error, {data: test_data_new, target: test_target_new, dropout: 1})
        print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * error))
#     incorrect = sess.run(error,{data: test_input, target: test_output})
# sess.close()
