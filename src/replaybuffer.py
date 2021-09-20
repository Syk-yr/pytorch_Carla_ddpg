"""
Data structure for implementing experience replay

"""
from collections import deque, namedtuple
import random
import numpy as np

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'terminal', 'next_state'])

class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=1234):
        self.max_size = buffer_size
        self.prt = 0
        self.target = False
        self.buffer = []
        random.seed(random_seed)
        # Right side of deque contains newest experience
        # self.buffer = deque()

    def push(self,data):
        if len(self.buffer) == self.max_size:
            self.buffer[int(self.prt)] = data
            self.prt = (self.prt + 1) % self.max_size
            self.target=True
        else:
            self.buffer.append(data)
    def size(self):
        return len(self.buffer)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.buffer), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.buffer[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

    def clear(self):
        self.buffer = []
        self.prt = 0

    # def add(self, state, action, reward, terminal, next_state):
    #     experience = Transition(state, action, reward, terminal, next_state)
    #     if self.count < self.buffer_size:
    #         self.buffer.append(experience)
    #         self.count += 1
    #     else:
    #         self.buffer.popleft()
    #         self.buffer.append(experience)

    # def size(self):
    #     return self.count

    # def sample_batch(self, batch_size):
    #     batch = []
    #     if self.count < batch_size:
    #         batch = random.sample(self.buffer, self.count)
    #     else:
    #         batch = random.sample(self.buffer, batch_size)

    #     return map(np.array, zip(*batch))

    # def clear(self):
    #     self.buffer.clear()
    #     self.count = 0


class ReplayBuffer2(object):
    def __init__(self, buffer_size, random_seed=1234):
        self.buffer_size = buffer_size
        self.count_positive = 0
        self.count_negative = 0
        self.buffer_positive = deque()
        self.buffer_negative = deque()
        random.seed(random_seed)

    def add(self, state, action, reward, terminal, next_state):
        experience = Transition(state, action, reward, terminal, next_state)
        if reward >= 0:
            if self.count_positive < self.buffer_size:
                self.buffer_positive.append(experience)
                self.count_positive += 1
            else:
                self.buffer_positive.popleft()
                self.buffer_positive.append(experience)
        else:
            if self.count_negative < self.buffer_size:
                self.buffer_negative.append(experience)
                self.count_negative += 1
            else:
                self.buffer_negative.popleft()
                self.buffer_negative.append(experience)

    def size(self):
        return self.count_negative + self.count_positive

    def sample_batch(self, batch_size):
        batch = []

