#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import tensorflow as tf

samples=66
texts=''
# build an vocabulary for the strings
for i in range(1,samples+1):
    path = './path/without dig/level ' + str(i)+'.txt'
    with open(path, 'r') as f:
        t=f.read()
        texts+=t
vocab = set(texts)
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
# encode each string to numbers according to the vocabulary   
encoded=[]
for i in range(1,samples+1):
    path = './path/without dig/level ' + str(i)+'.txt'
    with open(path, 'r') as f:
        text=f.read()
        encoded_text=np.array([vocab_to_int[c] for c in text])
        encoded=np.hstack((encoded,encoded_text))

#split strings to mini-batches
def get_batches(arr, batch_size, n_steps):
    
    batch_words = batch_size * n_steps
    n_batches = int(len(arr) / batch_words)
    arr = arr[:batch_words * n_batches]    
    arr = arr.reshape((batch_size, -1))    
    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n+n_steps]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

# build inputs for the model
def build_inputs(batch_size, num_steps):
    
    inputs = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')    
    return inputs, targets, keep_prob
# construct the LSTM model
def build_lstm(lstm_size, num_layers, batch_size, keep_prob):

    cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(lstm_size), output_keep_prob=keep_prob) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)    
    return cell, initial_state
# compute the probability  distribution of characters
def build_output(lstm_output, in_size, out_size):

    seq_output = tf.concat(lstm_output, axis=1)
    x = tf.reshape(seq_output, [-1, in_size])
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    logits = tf.matmul(x, softmax_w) + softmax_b
    out = tf.nn.softmax(logits, name='predictions')    
    return out, logits
# define loss function
def build_loss(logits, targets, lstm_size, num_classes):

    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)    
    return loss
# define gradient optimizer 
def build_optimizer(loss, learning_rate, grad_clip):

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))    
    return optimizer
# build CharRNN
class CharRNN:
    
    def __init__(self, num_classes, batch_size=28, num_steps=50, 
                       lstm_size=128, num_layers=2, learning_rate=0.001, 
                       grad_clip=5, sampling=False):

        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)
        # One-Hot encoding
        x_one_hot = tf.one_hot(self.inputs, num_classes)        
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)
        
# initial parameters
num_steps = 50          
lstm_size = 512         
num_layers = 2          
learning_rate = 0.001    
keep_prob = 0.5                
batch_size = 12        
epochs=60

model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers, 
                learning_rate=learning_rate)

saver = tf.train.Saver(max_to_keep=100)
# Train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    counter = 0
    for e in range(epochs):
        new_state = sess.run(model.initial_state)
        loss = 0
        for x, y in get_batches(encoded, batch_size, num_steps):
            counter += 1
            start = time.time()
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.loss, 
                                                 model.final_state, 
                                                 model.optimizer], 
                                                 feed_dict=feed)
            
            end = time.time()           
            if counter % 64 == 0:
                print('epoch: {}/{}'.format(e+1, epochs),
                      'steps: {}'.format(counter),
                      'loss: {:.4f}'.format(batch_loss),
                      '{:.4f} sec/batch'.format((end-start)))
            if (counter % 64 == 0):
                saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))    
    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))        
# randomly select a probability of a character        
def pick_top_n(preds, vocab_size, top_n=5):

    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c
# sample a character one by one according to the probability and join them as a string
def sample(checkpoint, n_samples, lstm_size, vocab_size, start="rrr"):

    samples = [c for c in start]
    model = CharRNN(vocab_size, lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:        
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in start:
            x = np.zeros((1, 1))
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

        c = pick_top_n(preds, vocab_size)
        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, vocab_size)
            samples.append(int_to_vocab[c])
        
    return ''.join(samples)     
        
checkpoint = tf.train.latest_checkpoint('checkpoints')
samp = sample(checkpoint, 100, lstm_size, len(vocab), start="lr")
# save the string       
with open("./trained_path/path1.txt","a")as the_file:
    for y in samp:
        the_file.write(y)
