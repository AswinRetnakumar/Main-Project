#!usr/bin/python
import gym

env = gym.make("CartPole-v0")
observation = env.reset()

for _ in range(2):
    
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print("Observation ")
    print(observation)
    print("\n")
    print("Reward")
    print(reward)
    print("\n)
