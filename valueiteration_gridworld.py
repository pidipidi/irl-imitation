import gym
import gym.spaces
import gym_gridworld

import numpy as np
import random

import pickle

class ValueIterationAgent(object):
    """ """
    def __init__(self, env, n_state, action_space, gamma=0.9, learning_rate=0.01):
        self.env      = env
        self.n_state  = n_state
        self.action_space = action_space
        self.v_table  = np.zeros(n_state) #value table
        
        self.gamma         = gamma # discount factor
        self.learning_rate = learning_rate

    def _create_trainsition_mtx(self):
        T = np.zeros((self.n_state, self.action_space.n, self.n_states))

        y0,x0 = np.unravel_index(range(self.n_states), self.env.grid_map_shape)

        T[self.env.pos2idx(y0,x0), : , self.env.pos2idx(y0,x0)] += 0.0

        action_prob = np.ones(4)*0.1
        action_prob[0,0] = 0.7
        action_prob[1,1] = 0.7
        action_prob[2,2] = 0.7
        action_prob[3,3] = 0.7

        for action in xrange(self.action_space.n):
            

        
        

    def act(self, state):
        """
        returns:
        next action
        """
        state_action = self.q_table[self.env.pos2idx(state)]

        max_v           = min(state_action)
        max_action_list = []
        for i, v in enumerate(state_action):
            if v > max_v:
                max_action_list = []
                max_v = v
                max_action_list.append(i)
            elif v == max_v:
                max_action_list.append(i)

        return random.choice(max_action_list)
                
    def learn(self, state, reward, action, state_next ):
        q = self.q_table[self.env.pos2idx(state), action]
        q_next = reward + self.gamma * max(self.q_table[self.env.pos2idx(state_next)])
        self.q_table[self.env.pos2idx(state),action] += self.learning_rate * (q_next - q)

    def save(self):
        pickle.dump( self.q_table, open( "valueiteration.pkl", "wb" ) )

    def load(self, filename=None):
        self.q_table = pickle.load( open( "valueiteration.pkl", "rb" ) )
    
def train(env, episode_count=1000):
    n_state = env.grid_map_shape[0]*env.grid_map_shape[1]
    ## epsilon = 0.1
    agent = ValueIterationAgent(env, n_state, env.action_space)

    for i in range(episode_count):
        state_now = env.reset()
        
        done = False
        
        for j in range(100):
            action = agent.act(state_now)
            state_next, reward, done, _ = env.step(action)

            # Learning new q table
            agent.learn(state_now, reward, action, state_next)

            state_now = state_next
            if done:
                print "Episode {} was successfull, Agent reached the goal".format(i)
                break
    
    agent.save()


def test(env):
    n_state = env.grid_map_shape[0]*env.grid_map_shape[1]
    agent = ValueIterationAgent(env, n_state, env.action_space)
    agent.load()

    episode_count = 10
    success_cnt   = 0

    for i in range(episode_count):
        state_now = env.reset()
        
        done = False
        
        for j in range(100):
            action = agent.act(state_now)
            state_next, reward, done, _ = env.step(action)
            state_now = state_next
            if done:
                success_cnt += 1
                print "Episode {} was successfull, Agent reached the goal".format(i)
                break

    print "Success Rate: ", success_cnt, " / ", episode_count

if __name__ == '__main__':

    env = gym.make('gridworld-v0')
    env.seed(0)
    env.verbose=True

    train(env, episode_count=1000)
    #test(env)
            
