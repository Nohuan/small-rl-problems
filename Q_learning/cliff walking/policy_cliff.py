import numpy as np
import gymnasium as gym
import time
def render_single(env, policy, max_steps=100):
    episode_reward = 0
    ob = env.reset()[0]  # Get the initial observation

    for t in range(max_steps):
        env.render()
        time.sleep(0.1)
        if isinstance(ob, np.ndarray):
            ob = ob.item()
        action = policy[ob]
        ob, reward, terminated, truncate, info = env.step(action)

        done  = terminated or truncate
        episode_reward += reward
        if done:
            break
    env.render()
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("\n"+"-"*50+f"\nEpisode reward: {episode_reward}\n"+"-"*50)


if __name__ == '__main__':
    qtables = np.load('outfile.npy')
    

    env = gym.make('CliffWalking-v0', render_mode='human')

    nS = env.observation_space.n
    

    q_last = qtables[-1,:,:]
    #print(q_last.shape) # (48,4)
    policy = np.zeros(nS, dtype = int)
    

    for state in range(nS):
        max_ids = np.where(q_last[state,:] == max(q_last[state,:]))[0]
        action  = np.random.choice(max_ids)
        policy[state] = action

    print(policy)

    render_single(env, policy)



    

    

