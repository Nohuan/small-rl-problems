import numpy as np
import gymnasium as gym
import time

from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

    P: nested dictionary
        From gym.core.Environment
        For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
        tuple of the form (probability, nextstate, reward, terminal) where
            - probability: float
                the probability of transitioning from "state" to "nextstate" with "action"
            - nextstate: int
                denotes the state we transition to (in range [0, nS - 1])
            - reward: int
                either 0 or 1, the reward for transitioning from "state" to
                "nextstate" with "action"
            - terminal: bool
              True when "nextstate" is a terminal state (hole or goal), False otherwise
    nS: int
        number of states in the environment
    nA: int
        number of actions in the environment
    gamma: float
        Discount factor. Number in range [0, 1)
"""

def pretty_print_frozen_lake(P, nS, nA):
    print("FrozenLake Environment Dynamics:")
    print("================================")

    for state in range(nS):
        print(f"\nState {state}:")
        for action in range(nA):
            action_name = ['LEFT', 'DOWN', 'RIGHT', 'UP'][action]
            print(f"  Action {action} ({action_name}):")
            for prob, next_state, reward, done in P[state][action]:
               status = "Goal" if reward == 1.0 else "Hole" if done else "Safe"
               print(f"Probability: {prob:.2f}, Next State: {next_state}, Reward: {reward:.2f}, Done: {done} ({status})")

def policy_evaluation(P, policy, value_function, gamma=0.9, tol=1e-3):
    while True:
        new_v_function = np.copy(value_function)
        for state, action in enumerate(policy):
            for prob, next_state, reward, done in P[state][action]:
                new_v_function[state] = prob * (reward + gamma * value_function[next_state])                
            value_change = np.sum(np.abs(value_function - new_v_function))
            value_function = new_v_function
        if value_change < tol:
            break
    return value_function

    
def policy_improvement(env,P, nS, nA, value_function, policy, gamma=0.9):
    new_policy = np.copy(policy)
    for state in range(nS):
        action_reward = []
        for action in range(nA):
            for prob, next_state, reward, done in P[state][action]:
                action_reward.append(prob * (reward + gamma * value_function[next_state]))
        new_policy[state] = np.argmax(action_reward)
    
    return new_policy
 

def policy_iteration(env,P, nS, nA, gamma=0.9,tol=10e-3):
    count = 0
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    multiple_policy = []
    while True:
        count +=1
        value_function = policy_evaluation(P, policy, value_function, gamma=0.9, tol=1e-3)
        new_policy = policy_improvement(env,P, nS, nA, value_function, policy, gamma=0.9)        
        policy_change = (new_policy != policy).sum()
        print(f'policy changed in {policy_change} states')
        multiple_policy.append(new_policy)
        policy = new_policy
        if policy_change == 0:
            break
    #print(type(policy))
    multiple_policy = np.array(multiple_policy)
    return value_function, policy , multiple_policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    while True:
        delta = 0
        for state in range(nS):        
            v = value_function[state]
            action_values = np.zeros(nA)
            for action in range(nA):
                for prob, next_state, reward, done in P[state][action]:
                    action_values[action] = prob * (reward + gamma * value_function[next_state])
            value_function[state] = np.max(action_values)
            delta = max(delta, abs(v - value_function[state]))
        if delta < tol:
            break

    #output a deterministic policy
    
    for state in range(nS):
        action_values = np.zeros(nA)
        for action in range(nA):
            for prob, next_state, reward, done in P[state][action]:
                action_values[action] = prob * (reward + gamma * value_function[next_state])
        policy[state] = np.argmax(action_values)

    
    return value_function, policy
def record_episodes(env,policy):
    obs, info = env.reset()
    #for episode_num in range()


def render_single(env, policy, max_steps=100):
    episode_reward = 0
    ob = env.reset()[0]  # Get the initial observation
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)
        if isinstance(ob, np.ndarray):
            ob = ob.item()
        a = policy[ob]
        ob, rew, done, _, _ = env.step(a)  # Unpack 5 values here
        episode_reward += rew
        if done:
            break
    env.render()
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("\n"+"-"*50+f"\nEpisode reward: {episode_reward}\n"+"-"*50)


if __name__ == '__main__':    
    env = gym.make('FrozenLake-v1',desc= None,map_name='4x4',is_slippery=False, render_mode='human')
    #para usar render_single se debe usar render_mode='human' y no 'rgb_array'
    #env = RecordVideo(env, video_folder='FrozenLakeVideos_New',name_prefix='eval',episode_trigger=lambda x: True)
    
    P = env.unwrapped.P
    nS = env.observation_space.n
    nA = env.action_space.n
    
    
    print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)
    V_pi, p_pi,multiple_policy = policy_iteration(env,P, nS, nA, gamma=0.9, tol=1e-3)  
    render_single(env, p_pi, 100)
    '''
    max_steps = 5
    step = 0
    obs, info = env.reset()        
    episode_over = False
    while not episode_over:
        action = multiple_policy[0][obs]
        obs, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
        if step >= max_steps:
            break
        step += 1
        
    env.close()
    '''      
        




    print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)
    #V_pi, p_pi = value_iteration(P, nS, nA, gamma=0.9, tol=1e-3)
    #render_single(env, p_pi, 100)


