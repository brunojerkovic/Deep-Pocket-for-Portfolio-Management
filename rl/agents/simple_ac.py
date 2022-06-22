import torch
from rl.agents.base_agent import Agent
from rl import Actor, Critic
import numpy as np
from utils.info_interfaces import InfoSender


class AgentSimpleAC(Agent, InfoSender):
    def __init__(self, *args, **kwargs):
        super(AgentSimpleAC, self).__init__(*args, **kwargs)

        self.actor = Actor(input_dims=self.input_dims,
                           lr=self.lr_a,
                           cpt_dir=self.cpt_dir,
                           device=self.device,
                           name='actor',
                           weight_init_method=self.weight_init_method)
        self.critic = Critic(input_dims=self.input_dims,
                             lr=self.lr_c,
                             cpt_dir=self.cpt_dir,
                             device=self.device,
                             name='critic',
                             weight_init_method=self.weight_init_method)

        self.actor_loss = None
        self.critic_loss = None
        self.actions = None

    def choose_action(self, state):
        # Perform an action
        self.actions = self.actor(state)
        self.actions = torch.max(self.actions, torch.full_like(self.actions, 10e-6))
        self.log_actions = torch.log(self.actions)
        return self.actions

    def learn(self, state_, reward, next_state_, done, gcn):
        # CRITIC Learning
        self.critic.optimizer.zero_grad()

        y_predicted = self.critic(state_).squeeze()
        y_expected = reward + self.gamma * self.critic(next_state_).squeeze()

        loss_critic = self.critic.loss(y_expected, y_predicted)
        loss_critic.backward(retain_graph=True)

        self.critic.optimizer.step()

        # ACTOR Learning
        self.actor.optimizer.zero_grad()

        #loss_actor = -1 * self.critic(state_).squeeze()
        #loss_actor_shaped = torch.full_like(self.actions, loss_actor.item())
        y_predicted = self.critic(state_).squeeze()
        y_expected = reward + self.gamma * self.critic(next_state_).squeeze()
        y_predicted_clone = y_predicted.clone().detach()
        y_expected_clone = y_expected.clone().detach()
        delta_loss = self.actor.loss(y_expected_clone, y_predicted_clone)
        loss_actor = (delta_loss * self.log_actions).mean()
        loss_actor.backward(retain_graph=True)
        #loss_actor_shaped = torch.full_like(self.actions, loss_actor.item())
        #self.actions.backward(loss_actor_shaped, retain_graph=True)

        self.actor.optimizer.step()

        self.actor_loss = np.float64(loss_actor.item())
        self.critic_loss = np.float64(loss_critic.item())

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def send_info(self) -> dict:
        return {
            'actor_loss': self.actor_loss,
            'critic_loss': self.critic_loss
        }

    def get_losses(self):
        return {'actor_loss': self.actor_loss, 'critic_loss': self.critic_loss}

    def get_models(self):
        return [self.actor, self.critic]
