import gym
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


env = gym.make("LunarLander-v2")

state_dim = 8
action_n = 4


class DeepCEM(nn.Module):
    def __init__(self, state_dim, action_n):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n
        self.network = nn.Sequential(nn.Linear(self.state_dim, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, self.action_n))
        self.softmax = nn.Softmax()
        self.optimazer = optim.Adam(self.parameters(), lr=1e-2)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, _input):
        return self.network(_input)
    
    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.forward(state)
        probs = self.softmax(logits).data.numpy()
        action = np.random.choice(self.action_n, p=probs)
        return action
    
    def fit(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                elite_states.append(state)
                elite_actions.append(action)
        elite_states = torch.FloatTensor(np.array(elite_states))
        elite_actions = torch.LongTensor(np.array(elite_actions))
        
        pred_actions = self.forward(elite_states)
        
        loss = self.loss(pred_actions, elite_actions)
        loss.backward()
        self.optimazer.step()
        self.optimazer.zero_grad()


def get_trajectory(env, agent, max_len=1000, visualize=False):
    trajectory = {'states': [], 'actions': [], 'rewards': []}
    
    state = env.reset()
    
    for _ in range(max_len):
        trajectory['states'].append(state)
        
        action = agent.get_action(state)
        trajectory['actions'].append(action)
        
        next_state, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)
        
        state = next_state
        
        if done:
            break
            
        if visualize:
            time.sleep(0.05)
            env.render()
    
    return trajectory


agent = DeepCEM(state_dim, action_n)

q_param = 0.6
iteration_n = 50
trajectory_n = 50


logger_str = 'Hyperparameters: iter_n {}, trajectory_n {}, q {}'.format(iteration_n, trajectory_n, q_param)
for iteration in tqdm(range(iteration_n), desc=logger_str):
    
    # Policy evaluation | Оценка политики
    trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
    total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
    
    # Policy Improvement | Совершенствование политики (отбор элитных траекторий)
    elite_trajectories = []
    quantile = np.quantile(total_rewards, q_param)
    for trajectory in trajectories:
        total_reward = np.sum(trajectory['rewards'])
        if total_reward > quantile:
            elite_trajectories.append(trajectory)
        
    # Обучение агента
    if len(elite_trajectories) > 0:
        agent.fit(elite_trajectories)


trajectory = get_trajectory(env, agent, max_len=1000, visualize=True)
print('Total reward:', sum(trajectory['rewards']))
