import gymnasium as gym

env = gym.make('FrozenLake-v1',render_mode='human',is_slippery=False)

actions = {
    'a':0,
    's':1,
    'd':2,
    'w':3}
observation, info = env.reset()
episode_over = False


while not episode_over:
    action = None
    while action not in actions:
        action = input('Usando wasd mueve al agente:')

    action = actions[action]

    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated
env.close()


        
