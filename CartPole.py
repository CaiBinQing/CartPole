import itertools

import gym
import matplotlib.pyplot as plt

from DQN import *

CHEAT = True


class CartPoleSolver:
    def __init__(self, env_name: str):
        self.env: gym.Env = gym.make(env_name)
        print("env:", self.env.spec.id)

        n_states = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        self.agent = DQN(n_states, n_actions)

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
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.plot(self.episode_reward, label="rewards")
        plt.plot(self.reward_means, label="mean")
        plt.hlines(self.env.spec.reward_threshold, 0, len(self.reward_means), label="target", color="red", linestyles="dashed")
        plt.legend(loc="upper left")
        plt.pause(0.0000001)  # pause a bit so that plots are updated

    def solve(self, target_update=10):
        env = self.env
        agent = self.agent

        for episode in itertools.count():
            state: np.array = env.reset()
            state: torch.Tensor = agent.preprocess_state(state)

            rewards = 0.  # 这一轮累计获得的奖励
            while True:
                # 执行一次动作
                action = agent.select_action(state)  # Tensor(1,1)
                next_state, reward, done, _ = env.step(action.item())  # (np.array, float, bool, {})
                rewards += reward
                if CHEAT:  # 手动分析特征 :(
                    cart_position, cart_velocity, pole_angle, pole_velocity = next_state
                    r1 = (env.x_threshold - abs(cart_position)) / env.x_threshold - 0.8  # reward 1: the closer the cart is to the center, the better
                    r2 = (env.theta_threshold_radians - abs(pole_angle)) / env.theta_threshold_radians - 0.5  # reward 2: the closer the pole is to the center, the better
                    reward = r1 + r2
                next_state = agent.preprocess_state(next_state)  # Tensor(1,n)
                # 记忆
                agent.store_transition(state, action, reward, None if done else next_state)
                # 训练
                agent.train()
                # 失败了就退出
                if done:
                    break
                # 切换到下一个状态
                state = next_state

            # 更新 target 网络
            if episode % target_update == 0:
                agent.update_target_net()
                print("episode:", episode, "reward:", rewards)

            # 记录
            self.log(rewards)
            self.draw_plot()

            # 检查是否已解决
            if self.isSolved():
                break

            # 测试一下
            if episode % 30 == 0:
                self.test(2)

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
                state, reward, done, _ = env.step(action.item())
                total_reward += reward
                state = agent.preprocess_state(state)
                if done:
                    break
        print('[Test] Evaluation Average Reward:', total_reward / count)


def random_test(env_name: str):
    env = gym.make(env_name)
    n_actions = env.action_space.n
    for episode in range(5):
        env.reset()
        for step in range(60):
            env.render()
            observation, reward, done, _ = env.step(random.randrange(n_actions))
            cart_position, cart_velocity, pole_angle, pole_velocity = observation


if __name__ == '__main__':
    ENV_NAME = "CartPole-v1"
    # random_test(ENV_NAME)
    solver = CartPoleSolver(ENV_NAME)
    solver.solve()
    # solver.draw_plot()
    solver.test(5)
    print("Done.")
