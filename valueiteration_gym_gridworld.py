import gym
import gym.spaces
import gym_gridworld

import numpy as np
import random

import copy
import pickle

class ValueIterationAgent(object):
    """ """
    def __init__(self, env, gamma=0.9, learning_rate=0.01):
        """
        inputs:
        env          Object  - Open AI Gym environment
        n_state      Integer - number of states (HxW)
        action_space Object  - ?
        rewards      ?
        gamma        float - RL discount
        error        float - threshold for a stop        
        
        """
        self.env      = env
        self.n_state  = env.grid_map_shape[0]*env.grid_map_shape[1]
        self.n_action = env.action_space.n
        ## self.action_space = action_space
        ## self.value_table  = np.zeros(n_state) #value table
        
        self.gamma         = gamma # discount factor
        self.learning_rate = learning_rate

        self.T      = self._create_trainsition_mtx()
        self.values = None
        self.policy = None

    def get_transition_mat(self):
        return copy.copy(self.T)

    def _create_trainsition_mtx(self):
        T = np.zeros((self.n_state, self.n_action, self.n_state))

        for i in range(self.n_state):
            for a1 in range(self.n_action):
                s = self.env.idx2pos(i)
                for a2 in range(self.n_action):
                    s_next = [s[0] + self.env.action_pos_dict[a2][0],
                              s[1] + self.env.action_pos_dict[a2][1]]

                    # Out of boundary
                    if s_next[0]<0 or s_next[0]>=self.env.grid_map_shape[0] or\
                      s_next[1]<0 or s_next[1]>=self.env.grid_map_shape[1]:    
                        continue

                    # Wall
                    if self.env.checkOnWALL(s_next):
                        continue

                    if a2 == 0:
                        T[i,a1,self.env.pos2idx(s_next)] = 0.0 # no action
                    elif a1 == a2:
                        T[i,a1,self.env.pos2idx(s_next)] = 0.7
                    else:
                        T[i,a1,self.env.pos2idx(s_next)] = 0.1
                        
        return T


    def value_iteration(self, error=0.01, deterministic=False):
        """
        inputs:
        """
        values = np.zeros([self.n_state])
        ## value_table_next = np.zeros(self.n_state) #value table

        # update each state
        while True:
            values_tmp = values.copy()
            
            for s in range(self.n_state):
                v_s = []
                values[s] = max([sum([self.T[s, a, s1]*(self.env.get_reward(self.env.idx2pos(s))+\
                                                        self.gamma*values_tmp[s1])\
                                                        for s1 in range(self.n_state)])\
                                                        for a in range(self.n_action)])

            if max([abs(values[s] - values_tmp[s]) for s in range(self.n_state)]) < error:
                break

        if deterministic:
            # generate deterministic policy
            policy = np.zeros([self.n_state])
            for s in range(self.n_state):
                policy[s] = np.argmax([sum([self.T[s, a, s1]*\
                                            (self.env.get_reward(self.env.idx2pos(s))+\
                                             self.gamma*values[s1])\
                                             for s1 in range(self.n_state)])\
                                             for a in range(self.n_action)])
        else:
            # generate stochastic policy
            policy = np.zeros([self.n_state, self.n_action])
            for s in range(self.n_state):
                v_s = np.array([sum([self.T[s, a, s1]*\
                                     (self.env.get_reward(self.env.idx2pos(s))+\
                                      self.gamma*values[s1]) \
                                      for s1 in range(self.n_state)]) \
                                      for a in range(self.n_action)])
                policy[s,:] = np.transpose(v_s/np.sum(v_s))

        self.policy = copy.copy(policy)
        self.values = copy.copy(values)
        return values, policy


    def act(self, state):
        """
        returns:
        next action
        """
        state_action = self.policy[self.env.pos2idx(state)] #multiple actions?
        if type(state_action) == np.float64:
            state_action = [state_action]
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

    
    ## def learn(self, state, reward, action, state_next ):
    ##     q = self.q_table[self.env.pos2idx(state), action]
    ##     q_next = reward + self.gamma * max(self.q_table[self.env.pos2idx(state_next)])
    ##     self.q_table[self.env.pos2idx(state),action] += self.learning_rate * (q_next - q)


    def save(self, filename="valueiteration.pkl"):
        pickle.dump( self.policy, open( filename, "wb" ) )


    def load(self, filename="valueiteration.pkl"):
        self.policy = pickle.load( open( filename, "rb" ) )

        
def train(env, episode_count=1000):
    state_now = env.reset()    
    agent = ValueIterationAgent(env)
    values, _  = agent.value_iteration(error=0.01, deterministic=False)

    agent.save()


def test(env):
    state_now = env.reset()    
    agent = ValueIterationAgent(env)
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

    train(env)
    test(env)
            
