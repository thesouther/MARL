import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time 
import threading

class RolloutWorker:
    def __init__(self, env, agents, conf):
        super().__init__()
        self.conf = conf
        self.agents = agents
        self.env = env
        self.episode_limit = conf.episode_limit
        self.n_actions = conf.n_actions
        self.n_agents = conf.n_agents
        self.state_shape = conf.state_shape
        self.obs_shape = conf.obs_shape

        self.start_epsilon = conf.start_epsilon
        self.anneal_epsilon = conf.anneal_epsilon
        self.end_epsilon = conf.end_epsilon
        print('Rollout Worker inited!')

    def generate_episode(self, episode_num=None, evaluate=False):
        if self.conf.replay_dir != '' and evaluate and episode_num == 0:
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        win_tag = False
        episode_reward = 0
        last_action = np.zeros((self.conf.n_agents, self.conf.n_actions))
        self.agents.policy.init_hidden(1)

        epsilon = 0 if evaluate else self.start_epsilon
        # if self.conf.epsilon_anneal_scale == 'episode':
        #     epsilon = epsilon - self.anneal_epsilon if epsilon > self.end_epsilon else epsilon
        # if self.conf.epsilon_anneal_scale == 'epoch':
        #     if episode_num == 0:
        #         epsilon = epsilon - self.anneal_epsilon if epsilon > self.end_epsilon else epsilon

        step = 0
        while not terminated and step < self.episode_limit:
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon, evaluate)

                # 生成动作的onehot编码
                action_onehot = np.zeros(self.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            # print("actions: ", actions)
            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            if self.conf.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.end_epsilon else epsilon

        # 最后一个动作
        o.append(obs)
        s.append(state)
        o_ = o[1:]
        s_ = s[1:]
        o = o[:-1]
        s = s[:-1]

        # target q 在last obs需要avail_action
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_ = avail_u[1:]
        avail_u = avail_u[:-1]

        # 当step<self.episode_limit时，输入数据加padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_.append(np.zeros((self.n_agents, self.obs_shape)))
            s_.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])
        
        episode = dict(
                    o=o.copy(),
                    s=s.copy(),
                    u=u.copy(),
                    r=r.copy(),
                    o_=o_.copy(),
                    s_=s_.copy(),
                    avail_u=avail_u.copy(),
                    avail_u_=avail_u_.copy(),
                    u_onehot=u_onehot.copy(),
                    padded = padded.copy(),
                    terminated = terminate.copy()
                )
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.start_epsilon = epsilon
        if evaluate and episode_num == self.conf.evaluate_epoch - 1 and self.conf.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        
        return episode, episode_reward, win_tag


class ReplayBuffer:
    def __init__(self, conf):
        self.conf = conf
        self.episode_limit = conf.episode_limit
        self.n_actions = conf.n_actions
        self.n_agents = conf.n_agents
        self.state_shape = conf.state_shape
        self.obs_shape = conf.obs_shape
        self.size = conf.buffer_size

        self.current_idx = 0
        self.current_size = 0

        self.buffers = {'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        'u': np.empty([self.size, self.episode_limit, self.n_agents, 1]),    
                        's': np.empty([self.size, self.episode_limit, self.state_shape]),    
                        'r': np.empty([self.size, self.episode_limit, 1]),    
                        'o_': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),  
                        's_': np.empty([self.size, self.episode_limit, self.state_shape]),    
                        'avail_u': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),    
                        'avail_u_': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),    
                        'u_onehot': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),    
                        'padded': np.empty([self.size, self.episode_limit, 1]),    
                        'terminated': np.empty([self.size, self.episode_limit, 1]),    
            }
        self.lock = threading.Lock()
        print("Replay Buffer inited!")

    def store_episode(self, episode_batch):
        batch_size = episode_batch['o'].shape[0] # 200
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            self.buffers['o'][idxs] = episode_batch['o']
            self.buffers['u'][idxs] = episode_batch['u']
            self.buffers['s'][idxs] = episode_batch['s']
            self.buffers['r'][idxs] = episode_batch['r']
            self.buffers['o_'][idxs] = episode_batch['o_']
            self.buffers['s_'][idxs] = episode_batch['s_']
            self.buffers['avail_u'][idxs] = episode_batch['avail_u']
            self.buffers['avail_u_'][idxs] = episode_batch['avail_u_']
            self.buffers['u_onehot'][idxs] = episode_batch['u_onehot']
            self.buffers['padded'][idxs] = episode_batch['padded']
            self.buffers['terminated'][idxs] = episode_batch['terminated']

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx+inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
