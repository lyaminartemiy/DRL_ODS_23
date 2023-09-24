import numpy as np
import gym
import gym_maze
import random
import time
import pygame


env = gym.make('maze-sample-5x5-v0')
state_n = 25
action_n = 4


class RandomAgent():
    def __init__(self, action_n):
        self.action_n = action_n
        pass
    
    def get_action(self, state):
        action = np.random.randint(self.action_n)
        return action
    
    
class CrossEntropyAgent():
    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((state_n, action_n)) / action_n
        
    def get_action(self, state):
        action = np.random.choice(np.arange(self.action_n), p=self.model[state])
        return int(action)
    
    def fit(self, elite_trajectories):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1
                
        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()
            
        self.model = new_model
        return None


def get_state(obs):
    return int(np.sqrt(state_n) * obs[0] + obs[1])


def get_trajectory(env, agent, max_len=1000, visulize=False):
    trajectory = {'states': [], 'actions': [], 'rewards': []}
    
    obs = env.reset()
    state = get_state(obs)
    
    for _ in range(max_len):
        trajectory['states'].append(state)
        
        action = agent.get_action(state)
        trajectory['actions'].append(action)
        
        obs, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)
        
        state = get_state(obs)
        
        if visulize:
            time.sleep(0.5)
            env.render()
        
        if done:
            break
    
    return trajectory


agent = CrossEntropyAgent(state_n, action_n)
q_param = 0.9
trajectory_n = 20
iteration_n = 50

for iteration in range(iteration_n):
    
    # Policy Evaluation
    trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
    total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
    print('Iteration:', iteration, 'mean total rewards:', np.mean(total_rewards))
    
    # Policy Improvment
    quantile = np.quantile(total_rewards, q_param)
    elite_trajectories = []
    for trajectory in trajectories:
        total_reward = np.sum(trajectory['rewards'])
        if total_reward > quantile:
            elite_trajectories.append(trajectory)
    
    agent.fit(elite_trajectories)
    


trajectory = get_trajectory(env, agent, max_len=1000, visulize=True)
print('Total reward:', sum(trajectory['rewards']))
print('Model:')
print(agent.model)