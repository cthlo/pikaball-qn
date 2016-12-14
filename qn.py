import config as cfg
from memory import Memory
from network import Network
from numpy.random import random, randint
import pikaball
import numpy as np

def epsilon(glob_step):
    return max(
        cfg.min_epsilon,
        cfg.init_epsilon - glob_step*cfg.epsilon_decrease_rate
    )

with Network() as net, Memory() as memory:
    obs_space_lo = pikaball.observation_space.low
    obs_space_hi = pikaball.observation_space.high
    out_of_bound = obs_space_lo - 1.0
    state_range = np.repeat(obs_space_hi - obs_space_lo, cfg.history_length)
    frame_size = cfg.frame_size

    for _ in range(cfg.train_episodes):
        history = np.tile(out_of_bound, cfg.history_length)
        history[-frame_size:] = pikaball.reset()

        action = 0
        prev_state = None
        frame_count = 1 # reset counts as first frame
        for _ in range(cfg.episode_maxframe):
            frame, reward, done, _ = pikaball.step(action)

            history[:-frame_size] = history[frame_size:]
            history[-frame_size:] = frame

            frame_count += 1
            if not done and frame_count % (cfg.skipframe+1) > 0:
                # skip frame
                continue
            elif history[0] == out_of_bound[0]:
                # state not filled
                assert not done, 'Episode too short'
                continue

            norm_state = [history*2/state_range-1]
            if prev_state:
                memory.add(prev_state, action, reward, norm_state)
            prev_state = norm_state

            qvals, action, glob_step = net.ffwd(norm_state)
            eps = epsilon(glob_step)
            if random() < eps:
                action = randint(cfg.num_actions)

            if memory.count > cfg.replay_start_count:
                pre_states, actions, rewards, pos_states = memory.sample(cfg.minibatch_size)
                net.train(pre_states, actions, rewards, pos_states)
                net.record_scalar('Epsilon', eps)

            labels = pikaball.action_space.labels
            print('\n'.join('%s : %f'%z for z in zip(labels, qvals[0]))+'\n')
            print(labels[action] + ' : ' + str(qvals[0, action]))
            print('Reward: %d' % reward)
            print('Epsilon: %f' % eps)
            print('--------------------------')

            if done:
                break
