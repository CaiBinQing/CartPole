import math
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> [Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    """
    current_states -> ... -> actions_value
    """

    def __init__(self, n_states: int, n_actions: int):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 12)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(12, n_actions)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


# Hyper Parameters
BATCH_SIZE = 64
GAMMA = 0.95  # 衰减因子
MEMORY_SIZE = 100000  # 记忆容量
TARGET_UPDATE = 30  # 每隔多少轮更新一次 target 网络
EPS_START = 1.  # 初始探索概率
EPS_END = 0.05  # 最终探索概率
EPS_DECAY = 200.  # 探索概率衰减参数


def get_epsilon(steps_done: int) -> float:
    return EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)


class DQN(object):
    """
    DQN agent
    """

    def __init__(self, n_states: int, n_actions: int, batch_size=BATCH_SIZE, gamma=GAMMA, target_update=TARGET_UPDATE, memory_size=MEMORY_SIZE):
        """
        :param n_states: input
        :param n_actions: output
        :param batch_size:
        :param gamma: 衰减因子
        :param target_update: 每隔多少轮更新一次 target 网络
        :param memory_size: 记忆容量
        """
        self.device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("use device:", self.device)

        self.n_actions = n_actions
        self.policy_net = Net(n_states, n_actions).to(self.device)
        self.target_net = Net(n_states, n_actions).to(self.device)
        self._update_target_net()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(memory_size)

        self.steps_done = 0

        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update

    def preprocess_state(self, state: np.array) -> Tensor:  # (1,n)
        return torch.from_numpy(state).unsqueeze(0).to(device=self.device, dtype=torch.float)

    def select_action(self, state: Tensor) -> int:
        """
        epsilon-greedy policy
        :param state: Tensor(1,n)
        """
        epsilon = get_epsilon(self.steps_done)
        self.steps_done += 1
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(dim=1).indices.item()

    def action(self, state: Tensor) -> int:
        """
        use target net
        :param state: Tensor(1,n)
        """
        with torch.no_grad():
            return self.policy_net(state).max(dim=1).indices.item()

    def finish_step(self, state: Tensor, action: int, reward: float, next_state: Tensor):
        """
        :param state: Tensor(1,n)
        :param action: int
        :param reward: float
        :param next_state: Tensor(1,n)
        """
        reward = torch.tensor([reward], device=self.device)  # Tensor(1,)
        action = torch.tensor([[action]], device=self.device)  # Tensor(1,1)
        self.memory.push(state, action, next_state, reward)
        self._train()

    def _train(self):
        if len(self.memory) < self.batch_size:
            return 0
        device = self.device

        transitions = self.memory.sample(self.batch_size)  # [Transition]
        batch = Transition(*zip(*transitions))  # [[state], [action], [next_state], [reward]]

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)  # Tensor(batch_size,)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])  # Tensor(partial_batch_size,n)
        state_batch = torch.cat(batch.state)  # Tensor(batch_size,n)
        action_batch = torch.cat(batch.action)  # Tensor(batch_size,1)
        reward_batch = torch.cat(batch.reward)  # Tensor(batch_size,)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
        # These are the actions which would've been taken for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)  # Tensor(batch_size,1)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based on the "older" target_net;
        # selecting their best reward with max(1)[0].
        # This is merged based on the mask,
        # such that we'll have either the expected state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)  # Tensor(batch_size,)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(dim=1).values.detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch  # Tensor(batch_size,)

        # Compute Huber loss
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def finish_episode(self, episode: int):
        # 更新 target 网络
        if episode % self.target_update == 0:
            self._update_target_net()

    def _update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
