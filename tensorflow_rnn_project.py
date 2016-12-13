import tensorflow as tf
import pandas as pd 
import functools
import numpy as np

COLUMNS = ["s680", "s720", "s760", "s800", "l680", "l720",
           "l760", "l800", "Normalized_O2_HHb", "Heart_Rate"]
FEATURES = ["s680", "s720", "s760", "s800", "l680", "l720",
           "l760", "l800", "Normalized_O2_HHb"]
feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]

tensorflow_dataframe = pd.read_csv("updated_tensorflow_csv_2.csv", skipinitialspace=True,
                            skiprows=0, names=COLUMNS)
print (tensorflow_dataframe)

import tensorflow as tf
tf.Graph().as_default()
tf.reset_default_graph() 

# vertical_stack_new = vertical_stack[[1,2,3,4,5,6,7,8,15,-3,-1]].astype(np.int)

data_features = tensorflow_dataframe.iloc[:, :9]
# print (len(data_features))
# print (data_features)
# print (vertical_stack_new.head(n=1))
data_predict = tensorflow_dataframe.iloc[:, 9:10]
# print (data_predict)
train_input = data_features.as_matrix()
train_output = data_predict.as_matrix()
print ("train_input.shape")
print (train_input.shape)

NUM_EXAMPLES = int(0.9 * len(train_input))
test_input_ = train_input[NUM_EXAMPLES:]
test_input = np.expand_dims(test_input_, axis=1)

test_output = test_output[NUM_EXAMPLES:] #everything beyond NUM_EXAMPLES

train_input_ = train_input[:NUM_EXAMPLES]
train_input = np.expand_dims(train_input_, axis=1)

train_output = train_output[:NUM_EXAMPLES] #till NUM_EXAMPLES
print ("train_input, test_input")
print (train_input.shape, test_input.shape)
print ("train_output, test_output")
print (train_output.shape, test_output.shape)

print ("test and training data loaded")

# max_length = int(target.get_shape()[1])
# print ("max length")
# print (max_length)
row_size = 1
data = tf.placeholder(tf.float32, [None, len(train_input[0]), 9]) #Number of examples, number of input, dimension of each input
# data = tf.placeholder(tf.float32, [None, 1, 8]) #Number of examples, number of input, dimension of each input

# choices = 220 #number of possible ouput classes - in this case it's the possible values of Heart Rate, including decimal values
choices = train_output.shape[1]
target = tf.placeholder(tf.float32, [None, 1])

# target = tf.reshape(target, [1, None, 1000])
print("data shape")
print (data.get_shape())
print ("target shape")
print (target.get_shape())

num_hidden = 10 #number of "neurons"
cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)

output, last_state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
print ("output shape")
print (output.get_shape())
output = tf.transpose(output, [1, 0, 2])
print (output.get_shape())

last = tf.gather(output, int(output.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[int(target.get_shape()[1])]))
         
print ("weight shape")
print (weight.get_shape())
print ("bias shape")
print (bias.get_shape())

# val = tf.reshape(val, [-1, num_hidden])

prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
print ("prediction shape")
print (prediction.get_shape())
# prediction = tf.reshape(prediction, [-1, max_length, out_size])

cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
# cross_entropy = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)

# mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
mistakes = tf.logical_or((tf.less(tf.argmax(target, 1), (tf.argmax(prediction, 1)-1))), \
            (tf.greater(tf.argmax(target, 1), (tf.argmax(prediction, 1)+1))))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

# Execute the model
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

# start a session and initialize all the variables defined to begin the training process

# experiment by changing batch size to see how it impacts your results and training time
batch_size = 7
no_of_batches = int(len(train_input)/batch_size)
epoch = 5
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    print ("Epoch - ",str(i))
incorrect = sess.run(error,{data: test_input, target: test_output})
print('Epoch {:2d} error {:31f}%'.format(i + 1, 100 * incorrect))
sess.close()

# print (sess.run(model.prediction,{data: []}))
