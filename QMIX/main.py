import os
import numpy as np
from smac.env import StarCraft2Env 
from agent import Agents
from utils import RolloutWorker, ReplayBuffer
import matplotlib.pyplot as plt

from config import Config
conf = Config()

def train():
    env = StarCraft2Env(
            map_name=conf.map_name,
            step_mul=conf.step_mul,
            difficulty=conf.difficulty,
            game_version=conf.game_version,
            replay_dir=conf.replay_dir
        )
    env_info = env.get_env_info() # {'state_shape': 61, 'obs_shape': 42, 'n_actions': 10, 'n_agents': 3, 'episode_limit': 200}
    conf.set_env_info(env_info)
    agents = Agents(conf)
    rollout_worker = RolloutWorker(env, agents, conf)
    buffer = ReplayBuffer(conf)

    # save plt and pkl
    save_path = conf.result_dir + conf.map_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    win_rates = []
    episode_rewards = []
    train_steps = 0
    for epoch in range(conf.n_epochs):
        # print("train epoch: %d" % epoch)
        episodes = []
        for episode_idx in range(conf.n_eposodes):
            episode, _, _ = rollout_worker.generate_episode(episode_idx)
            episodes.append(episode)

        episode_batch = episodes[0]
        episodes.pop(0)
        for episode in episodes:
            for key in episode_batch.keys():
                episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
     
        buffer.store_episode(episode_batch)
         # print(episode_batch['o'].shape)  # (1, 200, 3, 42)   
         # print(episode_batch['s'].shape)  # (1, 200, 61)      
         # print(episode_batch['u'].shape)  # (1, 200, 3, 1)    
         # print(episode_batch['r'].shape)  # (1, 200, 1)       
         # print(episode_batch['o_'].shape) # (1, 200, 3, 42)   
         # print(episode_batch['s_'].shape) # (1, 200, 61)      
         # print(episode_batch['avail_u'].shape)  # (1, 200, 3, 10)   
         # print(episode_batch['avail_u_'].shape) # (1, 200, 3, 10)   
         # print(episode_batch['padded'].shape)   # (1, 200, 1)       
         # print(episode_batch['terminated'].shape) # (1, 200, 1)  
        for train_step in range(conf.train_steps):
            mini_batch = buffer.sample(min(buffer.current_size, conf.batch_size)) # obs； (64, 200, 3, 42)
            # print(mini_batch['o'].shape)
            agents.train(mini_batch, train_steps)
            train_steps += 1

        if epoch % conf.evaluate_per_epoch == 0:
            win_rate, episode_reward = evaluate(rollout_worker)
            win_rates.append(win_rate)
            episode_rewards.append(episode_reward)
            print("train epoch: {}, win rate: {}%, episode reward: {}".format(epoch, win_rate, episode_reward))
            # show_curves(win_rates, episode_rewards)

    show_curves(win_rates, episode_rewards)

def evaluate(rollout_worker):
    # print("="*15, " evaluating ", "="*15)
    win_num = 0
    episode_rewards = 0
    for epoch in range(conf.evaluate_epoch):
        _, episode_reward, win_tag = rollout_worker.generate_episode(epoch, evaluate=True)
        episode_rewards += episode_reward
        if win_tag:
            win_num += 1
    return win_num / conf.evaluate_epoch, episode_rewards / conf.evaluate_epoch

def show_curves(win_rates, episode_rewards):
    print("="*15, " generate curves ", "="*15)
    plt.figure()
    plt.axis([0, conf.n_epochs, 0, 100])
    plt.cla()
    plt.subplot(2, 1, 1)
    plt.plot(range(len(win_rates)), win_rates)
    plt.xlabel('epoch*{}'.format(conf.evaluate_per_epoch))
    plt.ylabel("win rate")

    plt.subplot(2, 1, 2)
    plt.plot(range(len(episode_rewards)), episode_rewards)
    plt.xlabel('epoch*{}'.format(conf.evaluate_per_epoch))
    plt.ylabel("episode reward")

    plt.savefig(conf.result_dir + conf.map_name + '/result_plt.png', format='png')
    np.save(conf.result_dir + conf.map_name + '/win_rates', win_rates)
    np.save(conf.result_dir + conf.map_name + '/episode_rewards', episode_rewards)


if __name__ == "__main__":
    if conf.train:
        train()
        

