import gym
import tensorflow as tf
import numpy as np

num_inputs = 4
num_hiddenlayers = 4
num_outputs = 1 #Probability of actions

learning_rate = 0.01

initializer = tf.contrib.layers.variance_scaling_initializer()
X = tf.placeholder(tf.float32, shape=[None, num_inputs])
hidden_1 = tf.layers.dense(X, num_hiddenlayers,activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden_1, num_outputs)
out_layer = tf.nn.sigmoid(logits)

prob = tf.concat(axis= 1, values = [out_layer,1-out_layer])
action = tf.multinomial(prob, num_samples = 1)

y = 1- tf.to_float(action)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = y,logits = logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
gradients_and_variables = optimizer.compute_gradients(cross_entropy)

gradients = []
gradient_placeholders = []
grads_and_vars_feed = []

for gradient, variable in gradients_and_variables:
    gradients.append[gradient]
    gradient_placeholders = tf.placeholder(tf.float32, shape = gradient.get_shape())
    gradient_placeholders.append(gradient_placeholders)
    grads_and_vars_feed.append((gradient_placeholders, variable))

train_ops = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


