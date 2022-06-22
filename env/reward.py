import numpy as np
from utils.info_interfaces import InfoSender
from datasets.preprocessed_data import PreprocessedDataset


class Reward:
    def __init__(self, dataset: PreprocessedDataset, buffer_size, period,
                 p_0, c_b, c_s, mu_iters):
        self.dataset = dataset
        # self.closing_prices = Reward.get_closing_prices  # Key=date, values=closing prices of that date
        self.n_stocks = self.dataset.n_stocks
        self.v_prev = np.array([1] + [1] * self.n_stocks)
        self.rewards = [] # MyQueue(maxlen=reward_window)
        self.log_rewards = [] # MyQueue(maxlen=buffer_size)

        self.p_0 = p_0
        self.c_b = c_b
        self.c_s = c_s
        self.mu_iters = mu_iters
        self.cash = 1.
        self.zeros_x = None

        # Init parameters
        init_idx = buffer_size + dataset.date_to_idx(period[0])
        self.v_prev = np.concatenate(([self.cash], self.dataset.closing_prices[init_idx]))

    def get_reward(self, state_idx, actions):
        # Calculate current reward
        v_t = np.concatenate(([self.cash], self.dataset.closing_prices[state_idx]))
        y_t = v_t / self.v_prev
        current_reward = self.__calculate_mu(y_t, actions) * y_t @ actions
        log_reward = np.log(current_reward)

        # Save calculated values
        self.rewards.append(current_reward)
        self.log_rewards.append(log_reward)
        self.v_prev = v_t

        return current_reward

    def get_cum_portfolio_value(self):
        rewards_np = np.array(self.rewards)
        return self.p_0 * np.prod(rewards_np)

    def get_portfolio_value(self):
        rewards = np.array(self.rewards)
        p_t = rewards[-1] * rewards[-2] if len(rewards) > 1 else rewards[-1]
        return p_t

    def __calculate_mu(self, y_t, actions):
        def relu(x):
            self.zeros_x = self.zeros_x if self.zeros_x is not None else np.zeros_like(x)
            return np.maximum(self.zeros_x, x)
        # If there are no iterations of mu, then just don't use it
        if not self.mu_iters:
            return 1.

        actions_ = (y_t * actions) / (y_t @ actions)
        mu_prev = mu = 1 - 3 * self.c_b + self.c_s**2

        for _ in range(self.mu_iters):
            c1 = 1 / (1 - self.c_b * actions[0])
            c2 = 1 - self.c_b * actions_[0]
            c3 = -1 * (self.c_s + self.c_b - self.c_s * self.c_b)
            c4 = np.sum(relu(relu(actions_ - mu_prev * actions)))
            mu = c1 * (c2 + c3 * c4)

            # Convergence criterion
            if abs(mu - mu_prev) < 1e-5:
                break

        return 1. # TODO: (FOR FULL IMPL) try to return mu here
