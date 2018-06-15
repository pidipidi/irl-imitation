import gym
import gym.spaces
import gym_gridworld

import numpy as np
import random

import copy
import pickle

import valueiteration_gym_gridworld as vg
from maxent_irl import *


def generate_demonstrations(env, agent, n_trajs=100, len_traj=20, rand_start=False):

    trajs = []
    for i in range(n_trajs):
        if rand_start:
            # override start_pos
            start_pos = [np.random.randint(0, env.grid_map_shape[0]),
                         np.random.randint(0, env.grid_map_shape[1])]
            env.change_start_state(start_pos)
    
        state_now = env.reset()
        
        episode = []
        episode.append(Step(cur_state=env.pos2idx(state_now), action=None,
                            next_state=None, reward=None, done=False))
        
        # while not is_done:
        done = False
        for _ in range(len_traj-1):
            action = agent.act(state_now)
            state_next, reward, done, _ = env.step(action)        
            episode.append(Step(cur_state=env.pos2idx(state_now), action=action,
                                next_state=env.pos2idx(state_next), reward=reward, done=done))
            state_now = state_next
            if done:
                break

        if done:
            print "Episode {} was successfull, Agent reached the goal".format(i)            
            trajs.append(episode)

    return trajs

def feature_basis(env):
    from scipy.spatial import distance
    n_state  = env.grid_map_shape[0]*env.grid_map_shape[1]
    
    state_target = env.get_target_state()
    states = []
    for i in range(n_state):
        states.append(env.idx2pos(i))

    wall = []
    for i, state in enumerate(states):
        if env.checkOnWALL(state):
            wall.append(state)

    # rel pos from goal ----------------------------------------------------
    goal_hist_size = 10

    dists = []
    dists.append(distance.cdist(states, [state_target], metric='cityblock'))    
    ## dists = np.amax(dists)-dists

    feat_goal_dist = np.zeros((n_state, goal_hist_size))
    for i, dist_per_s in enumerate(dists):
        hist, _ = np.histogram(dist_per_s, goal_hist_size, range=(0,np.amax(dists)))
        feat_goal_dist[i] = hist

    # collision ------------------------------------------
    feat_collision = np.zeros((n_state,1))

    for i, state in enumerate(states):
        if state in wall:
            feat_collision[i][0] = 0.
        else:
            feat_collision[i][0] = 1.

    # done ------------------------------------------
    feat_done = np.zeros((n_state,1))

    for i, state in enumerate(states):
        if state == state_target:
            feat_done[i][0] = 1.
        else:
            feat_done[i][0] = 0.
            
    feat = np.hstack([feat_goal_dist, feat_collision, feat_done])
    print "Feature size: ", np.shape(feat)
    return feat
        

def plot(env, agent):
    import matplotlib.pyplot as plt
    plt.rcParams['image.cmap'] = 'jet'

    rmap = agent.get_rewards()
    
    
    plt.figure(figsize=(20,4))
    plt.subplot(1, 3, 1)
    img_utils.heatmap2d(rmap_gt, 'Rewards Map - Ground Truth', block=False)
    plt.subplot(1, 3, 2)
    img_utils.heatmap2d(np.reshape(values_gt, (H,W), order='F'), 'Value Map - Ground Truth', block=False)
    plt.subplot(1, 3, 3)
    img_utils.heatmap2d(np.reshape(path_gt, (H,W), order='F'), 'Path Map - Ground Truth', block=False)
    plt.show()
    sys.exit()
    
        
def train(env, episode_count=1000):
    state_now = env.reset()    
    agent = vg.ValueIterationAgent(env)
    ## values, _  = agent.value_iteration(error=0.01, deterministic=False)
    agent.load()

    # plot test!
    plot(env, agent)

    # generate demonstrations
    trajs = generate_demonstrations(env, agent, n_trajs=100, len_traj=100,
                                    rand_start=True)

    # feature selection
    feat_map = feature_basis(env)

    # run irl
    T = agent.get_transition_mat()
    T = np.swapaxes(T,1,2)
    rewards = maxent_irl(np.array(feat_map), T, gamma=0.95, trajs=trajs, lr=0.01, n_iters=20)
    print rewards

    # value iteration

    agent.save()
    


def test(env):
    state_now = env.reset()    
    agent = vg.ValueIterationAgent(env)
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
    env.verbose=False

    train(env)
    ## test(env)
            
