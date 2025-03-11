import gymnasium as gym
import numpy as np
from tqdm import tqdm
import pandas as pd

from typing import NamedTuple


from tempfile import TemporaryFile

import os



class Params(NamedTuple):
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    seed: int  # Define a seed so that we get reproducible results
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states

class Qlearning():
    def __init__(self,learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        td = (
            reward + self.gamma * np.max(self.qtable[new_state,:])
            - self.qtable[state,action]
        )
        q_update = self.qtable[state,action] + self.learning_rate * td
        return q_update
        
    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size,self.action_size))

class EpsilonGreedy:
    def __init__(self, epsilon, rng):
        self.epsilon = epsilon
        self.rng = rng

    def choose_action(self, action_space, state, qtable):
        random_prop = rng.uniform(0,1)
        if random_prop < self.epsilon:
            action = action_space.sample() # aleatoriamente
        else:
            max_ids = np.where(qtable[state, :] == max(qtable[state, :]))[0]
            action = np.random.choice(max_ids)
        return action
    
if __name__ == '__main__':
    #total_episode 100 y n_runs 2 para testear las salidas
    params = Params(
        total_episodes = 2000,
        learning_rate = 0.8,
        gamma = 0.95,
        epsilon = 0.1,
        seed = 123,
        n_runs = 20,
        action_size = None,
        state_size = None,
    )
    # seed esta es la nueva forma recomendada para crear numeros random
    rng = np.random.default_rng(params.seed)
    
    env = gym.make('CliffWalking-v0')
    params = params._replace(action_size=env.action_space.n)
    params = params._replace(state_size=env.observation_space.n)

    #test
    learner = Qlearning(
        learning_rate = params.learning_rate,
        gamma = params.gamma,
        state_size = params.state_size,
        action_size = params.action_size)

    explorer = EpsilonGreedy(
        epsilon = params.epsilon,
        rng = rng)


    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []

    for run in range(params.n_runs):  # 
        learner.reset_qtable()  # 
        for episode in tqdm(
            episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False
        ):
            state = env.reset()[0]  # Reset the environment
            step = 0
            done = False
            total_rewards = 0

            while not done:
                action = explorer.choose_action(
                    action_space = env.action_space, state = state, qtable = learner.qtable)
                all_states.append(state)
                all_actions.append(action)

                new_state, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated
                learner.qtable[state, action] = learner.update(
                    state, action, reward, new_state
                )
                total_rewards += reward
                step += 1

                # Our new state is state
                state = new_state

            rewards[episode, run] = total_rewards
            steps[episode, run] = step

        qtables[run, :, :] = learner.qtable


    #policy = qtables[-1, :, :]
    #print(qtables.shape)
    #print(qtables[-1, :, :].shape)

    #for state in range(env.observation_space.n):
    #    print(state,policy[state])

        
    #print(env.observation_space.n)


    # save to last q tables
    #outfile = TemporaryFile()
    np.save('outfile', qtables)


    print(os.path.abspath("."))


    print(f'Las qtables se han guardado en outfile')
