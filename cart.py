#!usr/bin/python
import gym
import tensorflow as tf
import numpy as np

num_inputs = 4
num_hiddenlayers = 4
num_outputs = 1 #Probability of actions

initializer = tf.contrib.layers.variance_scaling_initializer()
X = tf.placeholder(tf.float32, shape=[None, num_inputs])
hidden_1 = tf.layers.dense(X, num_hiddenlayers,activation=tf.nn.relu, kernel_initializer=initializer)
hidden_2 = tf.layers.dense(hidden_1, num_hiddenlayers,activation=tf.nn.relu, kernel_initializer=initializer)
out_layer = tf.layers.dense(hidden_2, num_outputs,activation=tf.nn.sigmoid, kernel_initializer=initializer)

prob = tf.concat(axis= 1, values = [out_layer,1-out_layer])
action = tf.multinomial(prob, num_samples = 1)
init = tf.global_variables_initializer()

step_limit = 500
ep = 50
env = gym.make("CartPole-v0")
avg_steps = []

with tf.Session() as sess:
    init.run()

    for i in range(ep):
        obs = env.reset()
        env.render()
        for step in range(step_limit):
            action_val = action.eval(feed_dict = {X:obs.reshape(1, num_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])

            if done:
                avg_steps.append(step)
                print("Done after {}steps".format(step))
                break
print("After {} episodes, avg steps per game: {}".format(ep,np.mean(avg_steps)))
env.close()
