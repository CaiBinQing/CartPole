import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, n_states: int, n_actions: int):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(n_states, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, n_actions)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


class Reinforce(object):
    def __init__(self, n_states: int, n_actions: int, gamma=0.99):
        self.device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("use device:", self.device)

        self.policy = Policy(n_states, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()

        self.gamma = gamma

    def preprocess_state(self, state: np.array) -> Tensor:  # (1,n)
        return torch.from_numpy(state).unsqueeze(0).to(device=self.device, dtype=torch.float)

    def select_action(self, state: Tensor) -> int:
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def action(self, state: Tensor) -> int:
        return self.select_action(state)

    def finish_step(self, state: Tensor, action: int, reward: float, next_state: Tensor):
        self.policy.rewards.append(reward)

    def finish_episode(self, episode: int):
        R = 0
        policy_loss = []
        returns = []
        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]
