import numpy as np
import torch
from policy import QMIX
from torch.distributions import Categorical

class Agents:
    def __init__(self, conf):
        self.conf = conf
        self.device = conf.device
        self.n_actions = conf.n_actions 
        self.n_agents = conf.n_agents 
        self.state_shape = conf.state_shape 
        self.obs_shape = conf.obs_shape
        self.episode_limit = conf.episode_limit

        self.policy = QMIX(conf)

        print("Agents inited!")

    def choose_action(self, obs, last_action, agent_num, availible_actions, epsilon, evaluate=False):
        inputs = obs.copy()
        # print(availible_actions)
        availible_actions_idx = np.nonzero(availible_actions)[0]
        agents_id = np.zeros(self.n_agents)
        agents_id[agent_num] = 1.

        if self.conf.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.conf.reuse_network:
            inputs = np.hstack((inputs, agents_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :]

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(self.device) # (42,) -> (1,42)
        availible_actions = torch.tensor(availible_actions, dtype=torch.float32).unsqueeze(0).to(self.device)

        # get q value
        q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_drqn_net(inputs, hidden_state)
        # choose action form q value
        q_value[availible_actions == 0.0] = -float("inf")
        if np.random.uniform() < epsilon:
            action = np.random.choice(availible_actions_idx)
        else:
            action = torch.argmax(q_value)
        return action

    def _get_max_episode_len(self, batch):
        terminated = batch["terminated"]
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx+1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):
        # 不同的episode的数据长度不同，因此需要得到最大长度
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.conf.save_frequency == 0:
            self.policy.save_model(train_step)

 