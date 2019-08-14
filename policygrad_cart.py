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
    gradients.append(gradient)
    gradient_placeholders = tf.placeholder(tf.float32, shape = gradient.get_shape())
    gradient_placeholders.append(gradient_placeholders)
    grads_and_vars_feed.append((gradient_placeholders, variable))

train_ops = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


def helper_discoutn_rewards(rewards, discount_rate):

    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards*discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_norm_rewards(all_rewards, discount_rate):

    all_discounted_rewards = []
    for rewards in all_rewards:
        all_discounted_rewards.append(helper_discoutn_rewards(rewards, discount_rate))
        
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean  = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/ reward_std for discounted_rewards in all_discounted_rewards]



env = gym.make("CartPole-v0")

rounds = 10
game_steps = 1000
iter = 700
discount_rate = 0.9

with tf.Session() as sess:
    sess.run(init)

    for i in range(iter):
        print("On iteration {}".format(iter))
        all_rewards = []
        all_gradients = []

        for step in range(rounds):

            current_reward = []
            current_gradient =[]

            obs = env.reset()

            for step in range(game_steps):

                action_val, gradient_val = sess.run([action,gradients], feed_dict= {})

                obs, reward, done, info = env.step(action_val[0][0])
                current_reward.append(reward)
                current_gradient.append(gradient_val)

                if done:
                    break
            all_rewards.append(current_reward)
            all_gradients.append(current_gradient)


    all_rewards = discount_norm_rewards(all_rewards, discount_rate)
    feed_dict =  {}
