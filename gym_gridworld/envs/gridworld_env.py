import gym
import sys
import os
import time
import copy
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from PIL import Image as Image
import matplotlib.pyplot as plt

EMPTY = 0
WALL  = 1
GOAL  = 3
START = 4

# define colors
# 0: black; 1 : gray; 2 : blue; 3 : green; 4 : red
COLORS = {EMPTY:[0.0,0.0,0.0], WALL:[0.5,0.5,0.5], \
          2:[0.0,0.0,1.0], GOAL:[0.0,1.0,0.0], \
          START:[1.0,0.0,0.0], 6:[1.0,0.0,1.0], \
          7:[1.0,1.0,0.0]}

class GridworldEnv(gym.Env):
    """
    We use row-major (C-style) order for state indices.
    """
    metadata = {'render.modes': ['human']}
    num_env = 0 
    def __init__(self, terminal_reward=1.0, wall_reward=0.0, step_reward=0.0):
        self.actions = [0, 1, 2, 3, 4]
        self.inv_actions = [0, 2, 1, 4, 3]
        self.action_space = spaces.Discrete(5)
        self.action_pos_dict = {0: [0,0], 1:[-1, 0], 2:[1,0], 3:[0,-1], 4:[0,1]}
 
        ''' set observation space '''
        self.obs_shape = [128, 128, 3]  # observation space shape
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape)

        ''' initialize system state ''' 
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        self.grid_map_path = os.path.join(this_file_path, 'dpark1.txt')        
        self.start_grid_map = self._read_grid_map(self.grid_map_path) # initial grid map
        self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map
        self.observation = self._gridmap_to_observation(self.start_grid_map)
        self.grid_map_shape = self.start_grid_map.shape

        ''' agent state: start, target, current state '''
        self.agent_start_state, _ = self._get_agent_start_target_state(
                                    self.start_grid_map)
        _, self.agent_target_state = self._get_agent_start_target_state(
                                    self.start_grid_map)
        self.agent_state = copy.deepcopy(self.agent_start_state)

        ''' set other parameters '''
        self.restart_once_done = False  # restart or not once done
        self.verbose = False # to show the environment or not
        self.seed()
        self.terminal_reward = terminal_reward
        self.wall_reward = wall_reward
        self.step_reward = step_reward
        self.is_done = False
        
        GridworldEnv.num_env += 1
        self.this_fig_num = GridworldEnv.num_env 
        if self.verbose == True:
            self.fig = plt.figure(self.this_fig_num)
            plt.show(block=False)
            plt.axis('off')
            self.render()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        ''' return next observation, reward, finished, success '''
        action = int(action)
        info = {}
        info['success'] = False
        self.is_done    = False
        nxt_agent_state = (self.agent_state[0] + self.action_pos_dict[action][0],
                            self.agent_state[1] + self.action_pos_dict[action][1])
        if action == 0: # stay in place
            info['success'] = True
            return (self.agent_state, self.step_reward, self.is_done, info)
        # Out of space
        if (nxt_agent_state[0] < 0 or nxt_agent_state[0] >= self.grid_map_shape[0]) or \
          (nxt_agent_state[1] < 0 or nxt_agent_state[1] >= self.grid_map_shape[1]):
            info['success'] = False
            return (self.agent_state, self.wall_reward, self.is_done, info)
        
        # successful behavior
        org_color = self.current_grid_map[self.agent_state[0], self.agent_state[1]]
        new_color = self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]]
        if new_color == EMPTY:
            if org_color == START:
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = EMPTY
                self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = START
            elif org_color == 6 or org_color == 7:
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = org_color-4 
                self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = 4
            self.agent_state = copy.deepcopy(nxt_agent_state)
        elif new_color == WALL: # gray
            info['success'] = False
            return (self.agent_state, self.wall_reward, self.is_done, info)
        elif new_color == 2 or new_color == GOAL:
            self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
            self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = new_color+4
            self.agent_state = copy.deepcopy(nxt_agent_state)
        self.observation = self._gridmap_to_observation(self.current_grid_map)
        self.render()

        # terminal state
        if nxt_agent_state[0] == self.agent_target_state[0] and nxt_agent_state[1] == self.agent_target_state[1] :
            target_observation = copy.deepcopy(self.observation)
            self.is_done = True
            if self.restart_once_done:
                self.observation = self.reset()
                info['success'] = True
                return (self.agent_state, self.terminal_reward, self.is_done, info)
            else:
                info['success'] = True
                return (self.agent_target_state, self.terminal_reward, self.is_done, info)
        else:
            info['success'] = True
            return (self.agent_state, self.step_reward, self.is_done, info)

    def reset(self):
        self.agent_state = copy.deepcopy(self.agent_start_state)
        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.observation = self._gridmap_to_observation(self.start_grid_map)
        self.render()
        return self.agent_state
        #return self.observation


    def get_reward(self, state):
        
        if state[0] == self.agent_target_state[0] and state[1] == self.agent_target_state[1]:
            return self.terminal_reward
        if self.current_grid_map[state[0], state[1]] == WALL:
            return self.wall_reward
        return self.step_reward
    

    def _read_grid_map(self, grid_map_path):
        grid_map = open(grid_map_path, 'r').readlines()
        grid_map_array = []
        for k1 in grid_map:
            k1s = k1.split(' ')
            tmp_arr = []
            for k2 in k1s:
                try:
                    tmp_arr.append(int(k2))
                except:
                    pass
            grid_map_array.append(tmp_arr)
        grid_map_array = np.array(grid_map_array)
        return grid_map_array

    def _get_agent_start_target_state(self, start_grid_map):
        start_state = None
        target_state = None
        for i in range(start_grid_map.shape[0]):
            for j in range(start_grid_map.shape[1]):
                this_value = start_grid_map[i,j]
                if this_value == START:
                    start_state = [i,j]
                if this_value == GOAL:
                    target_state = [i,j]
        if start_state is None or target_state is None:
            sys.exit('Start or target state not specified')
        return start_state, target_state

    def _gridmap_to_observation(self, grid_map, obs_shape=None):
        if obs_shape is None:
            obs_shape = self.obs_shape
        observation = np.random.randn(*obs_shape)*0.0
        gs0 = int(observation.shape[0]/grid_map.shape[0])
        gs1 = int(observation.shape[1]/grid_map.shape[1])
        for i in range(grid_map.shape[0]):
            for j in range(grid_map.shape[1]):
                for k in range(3):
                    this_value = COLORS[grid_map[i,j]][k]
                    observation[i*gs0:(i+1)*gs0, j*gs1:(j+1)*gs1, k] = this_value
        return observation
  
    def render(self, mode='human', close=False):
        if self.verbose == False:
            return
        img = self.observation
        fig = plt.figure(self.this_fig_num)
        plt.clf()
        plt.imshow(img)
        fig.canvas.draw()
        plt.pause(0.00001)
        return 
 
    def change_start_state(self, sp):
        ''' change agent start state '''
        ''' Input: sp: new start state '''
        if self.agent_start_state[0] == sp[0] and self.agent_start_state[1] == sp[1]:
            _ = self.reset()
            return True
        elif self.start_grid_map[sp[0], sp[1]] != EMPTY:
            return False
        else:
            s_pos = copy.deepcopy(self.agent_start_state)
            self.start_grid_map[s_pos[0], s_pos[1]] = EMPTY
            self.start_grid_map[sp[0], sp[1]] = START
            self.current_grid_map = copy.deepcopy(self.start_grid_map)
            self.agent_start_state = [sp[0], sp[1]]
            self.observation = self._gridmap_to_observation(self.current_grid_map)
            self.agent_state = copy.deepcopy(self.agent_start_state)
            self.reset()
            self.render()
        return True
        
    
    def change_target_state(self, tg):
        if self.agent_target_state[0] == tg[0] and self.agent_target_state[1] == tg[1]:
            _ = self.reset()
            return True
        elif self.start_grid_map[tg[0], tg[1]] != EMPTY:
            return False
        else:
            t_pos = copy.deepcopy(self.agent_target_state)
            self.start_grid_map[t_pos[0], t_pos[1]] = EMPTY
            self.start_grid_map[tg[0], tg[1]] = GOAL
            self.current_grid_map = copy.deepcopy(self.start_grid_map)
            self.agent_target_state = [tg[0], tg[1]]
            self.observation = self._gridmap_to_observation(self.current_grid_map)
            self.agent_state = copy.deepcopy(self.agent_start_state)
            self.reset()
            self.render()
        return True
    
    def get_agent_state(self):
        ''' get current agent state '''
        return self.agent_state

    def get_start_state(self):
        ''' get current start state '''
        return self.agent_start_state

    def get_target_state(self):
        ''' get current target state '''
        return self.agent_target_state

    def _jump_to_state(self, to_state):
        ''' move agent to another state '''
        info = {}
        info['success'] = True
        self.is_done    = False
        if self.current_grid_map[to_state[0], to_state[1]] == EMPTY:
            if self.current_grid_map[self.agent_state[0], self.agent_state[1]] == START:
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = EMPTY
                self.current_grid_map[to_state[0], to_state[1]] = START
                self.observation = self._gridmap_to_observation(self.current_grid_map)
                self.agent_state = [to_state[0], to_state[1]]
                self.render()
                return (self.observation, self.step_reward, self.is_done, info)
            if self.current_grid_map[self.agent_state[0], self.agent_state[1]] == 6:
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 2
                self.current_grid_map[to_state[0], to_state[1]] = 4
                self.observation = self._gridmap_to_observation(self.current_grid_map)
                self.agent_state = [to_state[0], to_state[1]]
                self.render()
                return (self.observation, 0, self.is_done, info)
            if self.current_grid_map[self.agent_state[0], self.agent_state[1]] == 7:  
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 3
                self.current_grid_map[to_state[0], to_state[1]] = 4
                self.observation = self._gridmap_to_observation(self.current_grid_map)
                self.agent_state = [to_state[0], to_state[1]]
                self.render()
                return (self.observation, 0, self.is_done, info)
        elif self.current_grid_map[to_state[0], to_state[1]] == START:
            return (self.observation, self.step_reward, self.is_done, info)
        elif self.current_grid_map[to_state[0], to_state[1]] == WALL:
            info['success'] = False
            return (self.observation, self.step_reward, self.is_done, info)
        elif self.current_grid_map[to_state[0], to_state[1]] == GOAL:
            self.current_grid_map[self.agent_state[0], self.agent_state[1]] = EMPTY
            self.current_grid_map[to_state[0], to_state[1]] = 7
            self.agent_state = [to_state[0], to_state[1]]
            self.observation = self._gridmap_to_observation(self.current_grid_map)
            self.render()
            self.is_done = True
            if self.restart_once_done:
                self.observation = self.reset()
                return (self.observation, self.terminal_reward, self.is_done, info)
            return (self.observation, self.terminal_reward, self.is_done, info)
        else:
            info['success'] = False
            return (self.observation, self.step_reward, self.is_done, info)

    def _close_env(self):
        plt.close(1)
        return
    
    def jump_to_state(self, to_state):
        a, b, c, d = self._jump_to_state(to_state)
        return (a, b, c, d) 


    def pos2idx(self, pos):
        """
        input:
        column-major 2d position
        returns:
        1d state index
        """
        return pos[1] + pos[0] * self.grid_map_shape[1]


    def idx2pos(self, idx):
        """
        input:
        1d state index
        returns:
        2d column-major position
        """
        return (idx / self.grid_map_shape[1], idx % self.grid_map_shape[1])

    
    def checkOnWALL(self, pos):

        if self.current_grid_map[pos[0], pos[1]] == WALL:
            return True
        else:
            return False 
