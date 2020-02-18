import itertools

import gym
import matplotlib.pyplot as plt

# from DQN import *
from Reinforce import *


class CartPoleSolver:
    def __init__(self, env_name: str):
        self.env: gym.Env = gym.make(env_name)
        print("env:", self.env.spec.id, "target:", self.env.spec.reward_threshold)

        n_states = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        # self.agent = DQN(n_states, n_actions)
        self.agent = Reinforce(n_states, n_actions)

        self.episode_reward: [float] = []  # 每一轮获得的奖励
        self.reward_means: [float] = []  # 均线

    def isSolved(self) -> bool:
        """
        CartPole-v0 is considered "solved" when the agent obtains
        an average reward of at least 195.0 over 100 consecutive episodes.
        """
        episode = len(self.reward_means)
        solved = episode >= 100 and self.reward_means[-1] >= self.env.spec.reward_threshold
        if solved:
            print("Solved after", episode - 100, "episodes.")
        return solved

    def log(self, reward: float):
        """
        :param reward: 这一轮获得的奖励
        """
        self.episode_reward.append(reward)
        self.reward_means.append(np.mean(self.episode_reward[-100:]))

    def draw_plot(self):
        plt.figure(self.env.spec.id)
        plt.clf()
        plt.title(self.env.spec.id)
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.plot(self.episode_reward, label="rewards")
        plt.plot(self.reward_means, label="mean")
        plt.hlines(self.env.spec.reward_threshold, 0, len(self.reward_means), label="target", color="red", linestyles="dashed")
        plt.legend(loc="upper left")
        plt.pause(0.0000001)  # pause a bit so that plots are updated

    def solve(self):
        env = self.env
        agent = self.agent

        for episode in itertools.count():
            state = env.reset()
            state = agent.preprocess_state(state)

            total_rewards = 0.  # 这一轮累计获得的奖励
            while True:  # step
                action = agent.select_action(state)  # int
                next_state, reward, done, _ = env.step(action)  # (np.array, float, bool, {})
                total_rewards += reward
                next_state = agent.preprocess_state(next_state)  # Tensor(1,n)
                agent.finish_step(state, action, reward, None if done else next_state)
                if done:
                    break
                state = next_state

            agent.finish_episode(episode)

            # log
            self.log(total_rewards)
            self.draw_plot()
            if episode % 10 == 0:
                print(f"Episode {episode:<6} Reward: {total_rewards:.0f}\tAverage: {self.reward_means[-1]:.2f}")

            # 检查是否已解决
            if self.isSolved():
                break

            # test
            if episode % 100 == 0:
                self.test(1)

    def test(self, count):
        env = self.env
        agent = self.agent

        total_reward = 0
        for i in range(count):
            state = env.reset()
            state = agent.preprocess_state(state)
            while True:
                env.render()
                action = agent.action(state)  # direct action for test
                state, reward, done, _ = env.step(action)
                total_reward += reward
                state = agent.preprocess_state(state)
                if done:
                    break
        print('Test Average Reward:', total_reward / count)


if __name__ == '__main__':
    ENV_NAME = "CartPole-v0"
    solver = CartPoleSolver(ENV_NAME)
    # solver.load()
    solver.solve()
    # solver.save()
    # solver.draw_plot()
    solver.test(5)
    print("Done.")
