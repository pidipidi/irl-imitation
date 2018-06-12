import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple


import img_utils
from mdp import gridworld
from mdp import value_iteration
from maxent_irl import *
from utils import *

Step = namedtuple('Step','cur_state action next_state reward done')


PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-hei', '--height', default=5, type=int, help='height of the gridworld')
PARSER.add_argument('-wid', '--width', default=5, type=int, help='width of the gridworld')
PARSER.add_argument('-g', '--gamma', default=0.8, type=float, help='discount factor')
PARSER.add_argument('-a', '--act_random', default=0.3, type=float, help='probability of acting randomly')
PARSER.add_argument('-t', '--n_trajs', default=100, type=int, help='number of expert trajectories')
PARSER.add_argument('-l', '--l_traj', default=20, type=int, help='length of expert trajectory')
PARSER.add_argument('--rand_start', dest='rand_start', action='store_true', help='when sampling trajectories, randomly pick start positions')
PARSER.add_argument('--no-rand_start', dest='rand_start',action='store_false', help='when sampling trajectories, fix start positions')
PARSER.set_defaults(rand_start=False)
PARSER.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='learning rate')
PARSER.add_argument('-ni', '--n_iters', default=20, type=int, help='number of iterations')
ARGS = PARSER.parse_args()
print ARGS


GAMMA = ARGS.gamma
ACT_RAND = ARGS.act_random
R_MAX = 1 # the constant r_max does not affect much the recoverred reward distribution
H = ARGS.height
W = ARGS.width
N_TRAJS = ARGS.n_trajs
L_TRAJ = ARGS.l_traj
RAND_START = ARGS.rand_start
LEARNING_RATE = ARGS.learning_rate
N_ITERS = ARGS.n_iters

R_MAX = 1.0
C_RWD = -10
EMPTY = -0.1


def feature_coord(gw):
  N = gw.height * gw.width
  feat = np.zeros([N, 2])
  for i in range(N):
    iy, ix = gw.idx2pos(i)
    feat[i,0] = iy
    feat[i,1] = ix
  return feat

def feature_basis(gw):
  """
  Generates a NxN feature map for gridworld
  input:
    gw      Gridworld
  returns
    feat    NxN feature map - feat[i, j] is the l1 distance between state i and state j
  """
  N = gw.height * gw.width
  feat = np.zeros([N, N])
  for i in range(N):
    for y in range(gw.height):
      for x in range(gw.width):
        iy, ix = gw.idx2pos(i)
        feat[i, gw.pos2idx([y, x])] = abs(iy-y) + abs(ix-x)
  return feat

def feature_histogram(gw):
    from scipy.spatial import distance
        
    N = gw.height * gw.width
    terminals = gw.get_terminals()
    ## M = 100 # number of features
    ## feat = np.zeros([N,M])
    states = np.zeros((N,2))
    for i in xrange(N):
        iy, ix = gw.idx2pos(i)
        states[i] = np.array([iy, ix])


    # rel pos from goal ----------------------------------------------------
    feat_goal_dist = np.zeros((N,2))
    for terminal in terminals:
        feat_goal_dist[:,0:1] += np.exp(-distance.cdist(states, np.array([[terminal[0],terminal[1]]]), metric='cityblock'))
        feat_goal_dist[:,1:] += distance.cdist(states, np.array([[0,0]]),
                                               metric='cityblock')

        
    # rel pos histogram from goal ------------------------------------------
    goal_hist_size = 10
    feat_goal = np.zeros((N,goal_hist_size))
    dists = []
    for terminal in terminals:
        dists.append(distance.cdist(states, np.array([[terminal[0],terminal[1]]]),
                                    metric='euclidean'))
    dists = np.array(dists)
    dists = np.swapaxes(dists, 0,1)
    
    for i, dist_per_s in enumerate(dists):
        hist, _ = np.histogram(dist_per_s, goal_hist_size, range=(0,np.sqrt(200)))
        feat_goal[i] = -hist
            
    # rel pos histogram -----------------------------------------------------
    obj_hist_size = 3
    feat_obj_dist = np.zeros((N,obj_hist_size))
    grid = gw.get_grid()
    objs = []
    for iy in xrange(len(grid)):
        for ix in xrange(len(grid[iy])):
            if grid[iy,ix] == C_RWD:
                objs.append((iy,ix))

    dists = []
    for obj in objs:
        dists.append(distance.cdist(states, np.array([obj]),
                                    metric='euclidean'))
    dists = np.array(dists)
    dists = np.swapaxes(dists, 0,1)

    for i, dist_per_s in enumerate(dists):
        hist, _ = np.histogram(dist_per_s, obj_hist_size, range=(0,np.sqrt(200)))
        feat_obj_dist[i] = hist

    # abs pos histogram -----------------------------------------------------
    pos_hist_size = 10
    feat_abs_pos = np.zeros((N,pos_hist_size*2))

    for i, state in enumerate(states):
        hist_x, _ = np.histogram(state[0], pos_hist_size, range=(0,10))
        hist_y, _ = np.histogram(state[1], pos_hist_size, range=(0,10))
        feat_abs_pos[i] = np.hstack([hist_x, hist_y])

    feat = np.hstack([feat_goal_dist, feat_obj_dist])
    #feat = np.hstack([feat_goal, feat_obj_dist, feat_abs_pos])
    print "Feature size: ", np.shape(feat)
    return feat


def generate_demonstrations(gw, policy, n_trajs=100, len_traj=20, rand_start=False, start_pos=[0,0]):
  """gatheres expert demonstrations

  inputs:
  gw          Gridworld - the environment
  policy      Nx1 matrix
  n_trajs     int - number of trajectories to generate
  rand_start  bool - randomly picking start position or not
  start_pos   2x1 list - set start position, default [0,0]
  returns:
  trajs       a list of trajectories - each element in the list is a list of Steps representing an episode
  """

  trajs = []
  for i in range(n_trajs):
    if rand_start:
      # override start_pos
      start_pos = [np.random.randint(0, gw.height), np.random.randint(0, gw.width)]

    episode = []
    gw.reset(start_pos)
    cur_state = start_pos
    cur_state, action, next_state, reward, is_done = gw.step(int(policy[gw.pos2idx(cur_state)]))
    episode.append(Step(cur_state=gw.pos2idx(cur_state), action=action, next_state=gw.pos2idx(next_state), reward=reward, done=is_done))
    # while not is_done:
    for _ in range(len_traj):
        cur_state, action, next_state, reward, is_done = gw.step(int(policy[gw.pos2idx(cur_state)]))
        episode.append(Step(cur_state=gw.pos2idx(cur_state), action=action, next_state=gw.pos2idx(next_state), reward=reward, done=is_done))
        if is_done:
            break
    trajs.append(episode)
  return trajs


def set_rewards():

  # init the gridworld
  # rmap_gt is the ground truth for rewards
  rmap_gt = np.zeros([H, W])
  rmap_gt[H-1, W-1] = R_MAX
  # rmap_gt[H-1, 0] = R_MAX

  rmap_gt[:,:] = 0
  rmap_gt[(H-1)/2,:] = C_RWD
  rmap_gt[(H-1)/2,5:7] = 0
  rmap_gt[(H-1)/2+1,7] = C_RWD
  rmap_gt[(H-1)/2+2,6] = C_RWD
  rmap_gt[(H-1)/2+3,5] = C_RWD
  #rmap_gt[(H-1)/2+4,4] = -10
  rmap_gt[H-1, W-1] = R_MAX


  for i in xrange(H):
      for j in xrange(W):
          if rmap_gt[i][j] == C_RWD or rmap_gt[i][j] == R_MAX:
              continue
          
          for ii in xrange(H):
              for jj in xrange(W):
                  if i==ii and j==jj: continue
                  
                  if rmap_gt[ii][jj] == C_RWD:
                      r = manhattan_dist((i,j), (ii,jj)) - manhattan_dist((i,j), (H-1,W-1))
                      if r < rmap_gt[i,j]:
                          rmap_gt[i,j] = r
  
  return rmap_gt

def set_rewards2():

  # init the gridworld
  # rmap_gt is the ground truth for rewards
  rmap_gt = np.zeros([H, W])+EMPTY
  rmap_gt[(H-1)/2,:]   = C_RWD
  rmap_gt[(H-1)/2,5:7] = EMPTY
  rmap_gt[(H-1)/2+1,7] = C_RWD
  rmap_gt[(H-1)/2+2,6] = C_RWD
  rmap_gt[(H-1)/2+3,5] = C_RWD
  #rmap_gt[(H-1)/2+4,4] = -10
  rmap_gt[H-1, W-1] = R_MAX

  return rmap_gt


def main():
  N_STATES  = H * W
  N_ACTIONS = 4

  rmap_gt = set_rewards2()
  gw = gridworld.GridWorld(rmap_gt, {(H-1,W-1)}, 1 - ACT_RAND)

  rewards_gt = np.reshape(rmap_gt, H*W, order='F')
  P_a = gw.get_transition_mat()

  values_gt, policy_gt = value_iteration.value_iteration(P_a, rewards_gt, GAMMA, error=0.01, deterministic=True)
  path_gt = gw.display_path_grid(policy_gt)

  ## #temp
  ## plt.figure(figsize=(20,4))
  ## plt.subplot(1, 3, 1)
  ## img_utils.heatmap2d(rmap_gt, 'Rewards Map - Ground Truth', block=False)
  ## plt.subplot(1, 3, 2)
  ## img_utils.heatmap2d(np.reshape(values_gt, (H,W), order='F'), 'Value Map - Ground Truth', block=False)
  ## plt.subplot(1, 3, 3)
  ## img_utils.heatmap2d(np.reshape(path_gt, (H,W), order='F'), 'Path Map - Ground Truth', block=False)
  ## plt.show()
  ## sys.exit()
  
  # use identity matrix as feature
  #feat_map = np.eye(N_STATES)

  # other two features. due to the linear nature, 
  # the following two features might not work as well as the identity.
  ## feat_map = feature_basis(gw)
  # feat_map = feature_coord(gw)
  feat_map = feature_histogram(gw)
  
  np.random.seed(1)
  trajs = generate_demonstrations(gw, policy_gt, n_trajs=N_TRAJS, len_traj=L_TRAJ,
                                  rand_start=RAND_START)
  rewards = maxent_irl(feat_map, P_a, GAMMA, trajs, LEARNING_RATE, N_ITERS)
  values, policy = value_iteration.value_iteration(P_a, rewards, GAMMA, error=0.01,
                                                   deterministic=True)


  path = gw.display_path_grid(policy)

  # plots
  plt.figure(figsize=(20,4))
  plt.subplot(2, 4, 1)
  img_utils.heatmap2d(rmap_gt, 'Rewards Map - Ground Truth', block=False)
  plt.subplot(2, 4, 2)
  img_utils.heatmap2d(np.reshape(values_gt, (H,W), order='F'), 'Value Map - Ground Truth', block=False)
  plt.subplot(2, 4, 3)
  img_utils.heatmap2d(np.reshape(rewards, (H,W), order='F'), 'Reward Map - Recovered', block=False)
  plt.subplot(2, 4, 4)
  img_utils.heatmap2d(np.reshape(values, (H,W), order='F'), 'Value Map - Recovered', block=False)
  plt.subplot(2, 4, 5)
  img_utils.heatmap2d(np.reshape(path_gt, (H,W), order='F'), 'Path Map - Ground Truth', block=False)
  plt.subplot(2, 4, 7)
  img_utils.heatmap2d(np.reshape(path, (H,W), order='F'), 'Path Map - Recovered', block=False)
  plt.show()

if __name__ == "__main__":
  main()
