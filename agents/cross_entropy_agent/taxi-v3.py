from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import itertools
import gym
import random
import time
import pygame
import warnings
warnings.filterwarnings('ignore')


env = gym.make('Taxi-v3')

state_n = 500  # Количество дискретных состояний
action_n = 6   # Количество уникальных действий


class CrossEntropyAgent():
    def __init__(self, state_n, action_n, stochastic_env=False, M=None):
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((state_n, action_n)) / action_n
        self.stochastic_env = stochastic_env
        self.M = M
        
    def get_action(self, state):
        p = self.model[state]
        p = p / np.sum(p)
        action = np.random.choice(np.arange(self.action_n), p=p)
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
    
    def fit_with_laplace(self, laplace_lambda, elite_trajectories):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1
                
        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                new_model[state] += laplace_lambda
                new_model[state] /= np.sum(new_model[state]) + laplace_lambda * action_n
            else:
                new_model[state] = self.model[state].copy()
                
        self.model = new_model
        return None
    
    def fit_with_policy_smoothing(self, policy_lambda, elite_trajectories):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1
                
        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()
                
        self.model = policy_lambda * new_model + (1 - policy_lambda) * self.model
        return None
    
    def fit_for_stochastic_env(self, iteration_n, trajectory_n):
        rewards = []
        for iteration in range(iteration_n):
            total_rewards = []
            
            if self.stochastic_env:
                stochastic_policy = agent.model.copy()
                trajectories = []
        
                for _ in tqdm(range(self.M), desc='Iteration {0}'.format(iteration)):
                    determ_policy = np.zeros((self.state_n, self.action_n))
                    
                    for state in range(self.state_n):
                        action = np.random.choice(np.arange(self.action_n), p=stochastic_policy[state])
                        determ_policy[state][action] += 1
                        
                    agent.model = determ_policy
                    trs = [get_trajectory(env, agent) for _ in range(trajectory_n)]
                    trajectories = trajectories + trs
                    local_reward = np.mean([np.sum(trajectory['rewards']) for trajectory in trs])
                    total_rewards += [local_reward for _ in trs]
                    
                rewards.append(np.mean(total_rewards))
                agent.model = stochastic_policy
                
        return rewards


def get_trajectory(env, agent, max_len=1000, visualize=False):
    
    # Запоминаем метрики траектории
    trajectory = {'states': [], 'actions': [], 'rewards': []}
    
    # Сброс состояния
    state = env.reset()
    
    # Итерируемся
    for _ in range(max_len):
        trajectory['states'].append(state)
        
        # Получаем действие
        action = agent.get_action(state)
        trajectory['actions'].append(action)
        
        # Делаем шаг, получаем награду и следующее состояние
        next_state, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)
        
        # Обновляем текущее состояние
        state = next_state
        
        if visualize:
            time.sleep(0.5)
            env.render()
        
        # Если завершили задачу, завершаем процесс
        if done:
            break
    
    return trajectory


def grid_search_agent_parametrs(iterations_grid, trajectories_grid, q_params_grid):
    # Пробегаемся по сеткам гиперпараметров
    for iteration_n in iterations_grid:
        for trajectory_n in trajectories_grid:
            for q_param in q_params_grid:

                # Инициализация агента
                agent = CrossEntropyAgent(state_n, action_n)

                logger_str = 'Hyperparameters: iter_n {}, trajectory_n {}, q {}'.format(iteration_n, trajectory_n, q_param)
                for iteraion in tqdm(range(iteration_n), desc=logger_str):

                    # Policy Evaluation | Оценка политики
                    trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
                    total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]

                    # Policy Improvement | Совершенствование политики (отбор элитных траекторий)
                    elite_trajectories = []
                    quantile = np.quantile(total_rewards, q_param)
                    for trajectory in trajectories:
                        total_reward = np.sum(trajectory['rewards'])
                        if total_reward > quantile:
                            elite_trajectories.append(trajectory)

                    # Фиксируем лучшие траектории и корректируем политику
                    agent.fit(elite_trajectories)

                trajectory = get_trajectory(env, agent, max_len=1000, visualize=False)
                print('Total reward:', sum(trajectory['rewards']))

    
iterations_grid = [20, 25, 30]  # Количество итераций 
trajectories_grid = [100, 200, 300, 400]  # Количество траекторий
q_params_grid = [0.5, 0.6, 0.7]  # Квантили

 
best_iter_n = 30
best_q_param = 0.6
best_trajectory_n = 400
global_rewards = []

# Инициализация агента
agent = CrossEntropyAgent(state_n, action_n)

logger_str = 'Hyperparameters: iter_n {}, trajectory_n {}, q {}'.format(best_iter_n, best_trajectory_n, best_q_param)
for iteraion in tqdm(range(best_iter_n), desc=logger_str):

    # Policy Evaluation | Оценка политики
    trajectories = [get_trajectory(env, agent) for _ in range(best_trajectory_n)]
    total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]

    # Policy Improvement | Совершенствование политики (отбор элитных траекторий)
    elite_trajectories = []
    quantile = np.quantile(total_rewards, best_q_param)
    for trajectory in trajectories:
        total_reward = np.sum(trajectory['rewards'])
        if total_reward > quantile:
            elite_trajectories.append(trajectory)
            
    global_rewards.append(total_rewards)

    # Фиксируем лучшие траектории и корректируем политику
    agent.fit(elite_trajectories)
    
    
def unlist_list(nested_list):
    return list(itertools.chain(*nested_list))


global_rewards_df = pd.DataFrame({'Reward': unlist_list(global_rewards),
                                  'Iteration': np.repeat(np.arange(best_iter_n), best_trajectory_n)})
  
sns.set(rc={'figure.figsize':(15, 10)})
sns.set_style("white")

plot = sns.boxplot(data=global_rewards_df, y='Reward', x='Iteration', color=".8", linewidth=.75, showfliers=False);
plot.set_xlabel("Iteration", size=15);
plot.set_ylabel("Reward", size=15);
title = "Agent training schedule\nbest_iter_n = {}; best_q_param = {}; best_trajectory_n = {} ".format(best_iter_n, best_q_param, best_trajectory_n)
plot.set_title(title, size=20);
plt.savefig("agent_training_schedule.png")

trajectory = get_trajectory(env, agent, max_len=1000, visualize=True)
print('Total reward:', sum(trajectory['rewards']))
