import torch
from rl.agents.base_agent import Agent
from rl import Actor, Critic
#from evaluation import EvalMetrics

class AgentSharpeAC(Agent):
    def __init__(self, *args, **kwargs):
        super(AgentSharpeAC, self).__init__(*args, **kwargs)

        self.actor = Actor(self.input_dims, self.lr_a, 'actor', self.lr_a, self.cpt_dir, self.device)
        self.critic = Critic(self.input_dims, 'critic', self.lr_c, self.cpt_dir, self.device)
        self.eval = EvalMetrics()

    def choose_action(self):
        state = self.buffer.tensor()
        actions = self.actor(state)
        self.log_actions = torch.log(actions)
        return actions

    def learn(self, state, reward, next_state, done):
        self.actor.optimizer.zero_grad()

        # Turn into tensors and put on a device
        state = torch.tensor(state).to(self.device)
        reward = torch.tensor(state).to(self.device)
        new_state = torch.tensor(state).to(self.device)

        # CRITIC
        self.critic.optimizer.zero_grad()

        y_expected = reward + self.gamma * self.critic(new_state)
        y_predicted = self.critic(state)

        loss_critic = self.critic.loss(y_predicted, y_expected)
        loss_critic.backward()

        self.critic.optimizer.step()

        # ACTOR
        self.actor.optimizer.zero_grad()

        actions = self.actor(state)
        loss_actor = -1 * self.eval.get_sharpe(weights=actions, closing_prices=state)
        loss_actor.backward()

        self.actor.optimizer.step()

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()