import gym.spaces

import sys
sys.path.append("/home/dpark/git/external/gym-gridworld")

import gym_gridworld
env = gym.make('gridworld-v0')
_ = env.reset()
_ = env.step(env.action_space.sample())


for i in xrange(1000):
    env.render()




