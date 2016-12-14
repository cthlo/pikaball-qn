from os import path
import pikaball

data_dir = path.join(path.dirname(__file__), 'models')
frame_size = pikaball.observation_space.shape()[0] # assuming 1-D observation

history_length = 2
state_size = history_length*frame_size
hidden_size = 10
num_actions = pikaball.action_space.n
train_episodes = 99999999999
episode_maxframe = 600 # 600 frames ~ 20 seconds in 30 fps
skipframe = 3 # obesrve every 4 frames
minibatch_size = 64
targ_net_update_freq = 10000 # update target network this many training steps
                             # 0 to not freeze network

# discount factor
gamma = 0.9

# exploration (linear annealing)
min_epsilon = 0.01
init_epsilon = 0.3
epsilon_decrease_rate = 2.9e-7 # 1m steps to min_epsilon

# learning rate (exponential decay)
min_alpha = 0.00001
init_alpha = 0.02
alpha_decay_steps = 2000
alpha_decay_rate = 0.96

# RMSProp momentum
momentum = 0.95

# Tensorboard
scat = lambda s, *v: s + '_' + '_'.join(str(i) for i in v)
hyper_dir = path.join(
    scat('history', history_length),
    scat('hidden', hidden_size),
    scat('skip', skipframe),
    scat('minibatch', minibatch_size),
    scat('net_update', targ_net_update_freq),
    scat('gamma', gamma),
    scat('epsilon', min_epsilon, init_epsilon, epsilon_decrease_rate),
    scat('alpha', min_alpha, init_alpha, alpha_decay_steps, alpha_decay_rate),
    scat('momen', momentum)
)
summary_folder = path.join(data_dir, hyper_dir)

# Replay memory
memory_file = path.join(data_dir, 'memory.npy')
memory_stat_file = path.join(data_dir, 'memory_stat.npy')
memory_size = 200000
memory_save_freq = 10000
replay_start_count = 10000
