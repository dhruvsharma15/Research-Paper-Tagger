# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 18:08:38 2018

@author: dhruv
"""

import numpy as np
import tensorflow as tf
import pickle
import time
start = time.perf_counter()
import os

###### this data has been vectorized, that is, processed to be containing the word vectors
data_path = 'vector_data/'
label_path = 'subsampled_labels.txt'
#empty_vector = np.random.rand(300)

def get_batches(x, y, batch_size=100):
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]
        
def read_data(ind):
    papers = os.listdir(data_path)[ind*3:(ind+1)*3]
    data = []
    count = 0
    for p in papers:
        print('reading batch no',str(ind*3+1+count))
        count+=1
        article_path = data_path+p
        with open(article_path, 'rb') as f:
            d = pickle.load(f)
        data.extend(d)
    return data

#Current set of Hyperparmeters
lstm_size = 512
lstm_layers = 1
batch_size = 256
n_epochs = 50
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
#loss = tf.reduce_mean(tf.square(Y - predictions))
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = predictions))
optimizer = tf.train.AdamOptimizer().minimize(loss)	

#Accuracy
correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.float32), Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#Start Time
start_time = time.time()

#Temp Variables
x_ = [] #Accounts for batch size
loss_count = 0
train_loss = []
val_loss = []

ind = [0,1,2]
for i in ind:
    print('reading data')
    train_x = read_data(i)
    train_x = np.array(train_x)

    with open(label_path, 'rb') as f:
        subsampled_labels = pickle.load(f)

    train_y = subsampled_labels[i*3000:i*3000+len(train_x)]
    train_y = np.array(train_y)


#START SESSION:
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

#Serialisation vars:
    saver = tf.train.Saver(tf.global_variables())
    if(i!=0):
        ckpt = tf.train.get_checkpoint_state("saved_model/")
        saver.restore(sess, ckpt.model_checkpoint_path)

    print('training started')
    count_=0
#For every Epoch
    for e in range(n_epochs):
        batch_index = 1 #represents index of batch
	#For every batch
        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            if(count_==0):
                state = sess.run(initial_state, feed_dict =  {X: x, Y: y})
            feed = {X: x, Y: y, initial_state: state}
            state, loss_,  _ = sess.run([final_state, loss, optimizer], feed_dict=feed)
    #represents index of batch
            batch_index +=1
            count_+=1
            if(count_%5==0):
                print(loss_)
            if count_ % 10 == 0:
                hours, rem = divmod(time.perf_counter() - start, 3600)
                minutes, seconds = divmod(rem, 60)
                saver.save(sess, "saved_model/model.ckpt", global_step=count_)
                print(" Epoch {0}: Model is saved.".format(batch_index // 5),
                "Elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds) , "\n")
		#All epochs completed
        print('Training Completed for iteration',e)



