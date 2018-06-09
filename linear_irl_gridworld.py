import numpy as np
import matplotlib.pyplot as plt
import argparse

import img_utils
from mdp import gridworld
from mdp import value_iteration
from lp_irl import *

PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-l', '--l1', default=10, type=float, help='l1 regularization')
PARSER.add_argument('-g', '--gamma', default=0.5, type=float, help='discount factor')
PARSER.add_argument('-r', '--r_max', default=10, type=float, help='maximum value of reward')
PARSER.add_argument('-a', '--act_random', default=0.3, type=float, help='probability of acting randomly')
ARGS = PARSER.parse_args()
print ARGS


GAMMA = ARGS.gamma
ACT_RAND = ARGS.act_random
R_MAX = ARGS.r_max
L1 = ARGS.l1


def main():
  """
  Recover gridworld reward using linear programming IRL
  """

  H = 10
  W = 10
  N_STATES = H * W
  N_ACTIONS = 5

  # init the gridworld including the reward
  grid = [['0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],
          ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],
          ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],
          ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],
          ['-1', '-1', '-1', '-1', '-1', '0', '0', '-1', '-1', '-1'],
          ## ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],
          ['0', '0', '0', '0', '0', '0', '0', '-1', '0', '0'],
          ['0', '0', '0', '0', '0', '0', '-1', '0', '0', '0'],
          ['0', '0', '0', '0', '0', '-1', '0', '0', '0', '0'],
          ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],
          ['0', '0', '0', '0', '0', '0', '0', '0', '0', str(R_MAX)]]

  # custom
  for i, row in enumerate(grid):
      for j, e in enumerate(row):
          if e is '0':
              grid[i][j] = '-1'
          elif e is '-1':
              grid[i][j] = '-10'

  # grid, terminal state, trans_prob
  gw = gridworld.GridWorld(grid, {(H - 1, W - 1)}, 1 - ACT_RAND)

  # solve the MDP using value iteration
  vi = value_iteration.ValueIterationAgent(gw, GAMMA, 100)
  r_mat_gt = gw.get_reward_mat()
  v_mat_gt = gw.get_values_mat(vi.get_values())

  # Construct transition matrix
  P_a = np.zeros((N_STATES, N_STATES, N_ACTIONS))

  for si in range(N_STATES):
    statei = gw.idx2pos(si)
    for a in range(N_ACTIONS):
      probs = gw.get_transition_states_and_probs(statei, a)
      for statej, prob in probs:
        sj = gw.pos2idx(statej)
        # Prob of si to sj given action a
        P_a[si, sj, a] = prob

  # display policy and value in gridworld just for debug use
  gw.display_policy_grid(vi.get_optimal_policy())
  gw.display_value_grid(vi.values)

  # display a path following optimal policy
  ## print 'show optimal path. any key to continue'
  path_gt = gw.display_path_grid(vi.get_optimal_policy())
  ## img_utils.heatmap2d(np.reshape(path, (H, W), order='F'), 'Path')
  ## sys.exit()


  # setup policy
  policy = np.zeros(N_STATES)
  for i in range(N_STATES):
    policy[i] = vi.get_action(gw.idx2pos(i))

  #------------------ After getting optimal policy through iterations ------------------
  # solve for the rewards
  rewards = lp_irl(P_a, policy, gamma=GAMMA, l1=L1, R_max=R_MAX)
  r_mat = np.reshape(rewards, (H, W), order='F')
  v_mat = gw.get_values_mat(vi.get_values())
  path  = gw.display_path_grid(vi.get_optimal_policy())

  # display recoverred rewards
  print 'show recoverred rewards map. any key to continue'
  ## img_utils.heatmap2d(np.reshape(rewards, (H, W), order='F'), 'Reward Map - Recovered')
  #img_utils.heatmap3d(np.reshape(rewards, (H, W), order='F'), 'Reward Map - Recovered')

  # display a path following optimal policy
  print 'show optimal path. any key to continue'
  ## path = gw.display_path_grid(vi.get_optimal_policy())
  ## img_utils.heatmap2d(np.reshape(path, (H, W), order='F'), 'Path')


  # plots
  plt.figure(figsize=(20,4))
  plt.subplot(2, 4, 1)
  img_utils.heatmap2d(r_mat_gt, 'Rewards Map - Ground Truth', block=False)
  plt.subplot(2, 4, 2)
  img_utils.heatmap2d(np.reshape(v_mat_gt, (H,W), order='F'), 'Value Map - Ground Truth', block=False)
  plt.subplot(2, 4, 3)
  img_utils.heatmap2d(np.reshape(r_mat, (H,W), order='F'), 'Reward Map - Recovered', block=False)
  plt.subplot(2, 4, 4)
  img_utils.heatmap2d(np.reshape(v_mat, (H,W), order='F'), 'Value Map - Recovered', block=False)

  plt.subplot(2, 4, 5)
  img_utils.heatmap2d(np.reshape(path_gt, (H,W), order='F'), 'Path Map - Ground Truth', block=False)
  plt.subplot(2, 4, 7)
  img_utils.heatmap2d(np.reshape(path, (H,W), order='F'), 'Path Map - Recovered', block=False)
  
  plt.show()
  

if __name__ == "__main__":
  main()
