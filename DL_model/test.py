# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:35:47 2018

@author: dhruv
"""


import numpy as np
import tensorflow as tf
import pickle
import time
start = time.perf_counter()
import os
import pickle

###### this data has been vectorized, that is, processed to be containing the word vectors
data_path = 'vector_data/'
label_path = 'subsampled_labels.txt'


def get_batches(x, y, batch_size=100):
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]
        
def read_data(ind):
    papers = os.listdir(data_path)[9:10]
    data = []
    count = 0
    for p in papers:
        count+=1
        article_path = data_path+p
        with open(article_path, 'rb') as f:
            d = pickle.load(f)
        data.extend(d)
    return data

#Current set of Hyperparmeters
lstm_size = 512
lstm_layers = 1
output_dim = 32
	
from tensorflow.python.framework import ops
ops.reset_default_graph()

print('generating LSTM-network graph')
#Placeholder
X = tf.placeholder(tf.float32, [None, None, 300], name = 'inputs')
Y = tf.placeholder(tf.float32, [None, output_dim], name = 'labels')

#Build Network
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
cell = tf.contrib.rnn.MultiRNNCell([lstm]*lstm_layers)
	
initial_state = cell.zero_state(tf.shape(X)[0] , tf.float32)
outputs, final_state = tf.nn.dynamic_rnn(cell, X, initial_state = initial_state)
predictions = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=tf.nn.sigmoid)

#Optimisation
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = predictions))
optimizer = tf.train.AdamOptimizer().minimize(loss)	

#Accuracy
correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.float32), Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#Start Time
start_time = time.time()

#Temp Variables

ind = [0,1,2]

print('reading data')
test_x = read_data(9)
test_x = np.array(test_x)

with open(label_path, 'rb') as f:
    subsampled_labels = pickle.load(f)

test_y = subsampled_labels[9000:10000]
test_y = np.array(test_y)
print('test_y shape:',test_y.shape)
print('test_x shape:',test_x.shape)

#START SESSION:
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Serialisation vars:
saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state("saved_model/")
saver.restore(sess, ckpt.model_checkpoint_path)

tag_predictions = []

feed = {X:test_x, Y: test_y}
pred = sess.run( [predictions], feed_dict=feed)

tag_predictions.extend(pred)
    
with open('output.pkl','wb') as f:
    pickle.dump(pred[0], f)

print('prediction shape:',len(pred[0]),len(pred[0][0]))
 
with open('actual.pkl','wb') as f1:
    pickle.dump(test_y, f1)



