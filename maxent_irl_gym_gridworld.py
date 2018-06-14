import gym
import gym.spaces
import sys
## sys.path.append("/home/dpark/git/external/gym-gridworld")
import gym_gridworld

import numpy as np

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

class CustomAgent(object):
    """ """
    def __init__(self, observation_space, action_space):
        #self.observation_space = observation_space
        self.action_space = action_space

    def act(self, state, reward, done):
        #Q = np.zeros([self.observation_space.n, self.action_space.n])
        
        return self.action_space.sample()
        ## return np.argmax(Q[state])

def random_motion(env):
    agent = RandomAgent(env.action_space)
    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
    
if __name__ == '__main__':

    env = gym.make('gridworld-v0')
    env.seed(0)
    env.verbose=True

    # 1. random motion test
    ## random_motion(env)
    
    # 2. RL?
    episode_count = 1
    epsilon = 0.1
    gamma = 0.9
    
    for i in range(episode_count):
        state_now = env.reset()
        agent = CustomAgent(env.observation_space, env.action_space)
        reward = 0
        done = False
        
        for j in range(100):
            action = agent.act(state_now, reward, done)
            state_next, reward, done, _ = env.step(action)

            state_now = state_next
            if done:
                print "Episode {} was successfull, Agent reached the goal".format(i)
                break

    
    ## rewards_gt = np.reshape(rmap_gt, H*W, order='F')
    ## P_a = gw.get_transition_mat()
    ## values_gt, policy_gt = value_iteration.value_iteration(P_a, rewards_gt, GAMMA, error=0.01, deterministic=True)
    ## path_gt = gw.display_path_grid(policy_gt)

    ## rmap_gt = gw.get_reward_mat()


    # Close the env and write monitor result info to disk
    ## env.close()
    
    ## _ = env.reset()
    ## _ = env.step(env.action_space.sample())


    ## for i in xrange(10):
    ##     env.render()

