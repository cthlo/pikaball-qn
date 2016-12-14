import config as cfg
from numpy.random import randint
import numpy as np

class Memory(object):
    def __init__(self):
        try:
            self.mem = np.load(cfg.memory_file)
            self.cursor, self.count = np.load(cfg.memory_stat_file)
            print('Memory loaded')
        except IOError:
            # (pre_state, action, reward, pos_state)
            self.mem = np.empty([cfg.memory_size, cfg.state_size*2+2])
            self.cursor = self.count = 0
            print('New memory db')

    def add(self, pre_state, action, reward, pos_state):
        self.mem[self.cursor, :] = np.concatenate([
            pre_state, [[action]], [[reward]], pos_state
        ], 1)
        self.count = max(self.count, self.cursor + 1)
        self.cursor = (self.cursor + 1) % cfg.memory_size

        if self.cursor % cfg.memory_save_freq == 0:
            self.__backup()

    def sample(self, num):
        rows = [randint(self.count) for _ in range(cfg.minibatch_size)]
        trans = self.mem[rows, :]
        pre_states = trans[:, :cfg.state_size]
        actions = trans[:, cfg.state_size]
        rewards = trans[:, cfg.state_size+1]
        pos_states = trans[:, -cfg.state_size:]
        return pre_states, actions, rewards, pos_states

    def __backup(self):
        np.save(cfg.memory_file, self.mem)
        np.save(cfg.memory_stat_file, (self.cursor, self.count))
        print('Memory saved')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.__backup()
