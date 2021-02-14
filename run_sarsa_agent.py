from auxFunctions import getState, load_obj, maxAction
import gym

env = gym.make('MountainCar-v0')
# Set the max step to 200
env._max_episode_steps = 200

# Load Q-table trained with SARSA
Q = load_obj('pre-trained-SARSA')

# Run 10 episodes
for episode in range(10):
    done = False
    observation = env.reset()
    state = getState(observation)
    # While the car don't reach the goal or number of steps < 200
    while not done:
        env.render()
        print(observation)
        # Take the best action for that state given trained values
        action = maxAction(Q, state)
        observation, reward, done, info = env.step(action)
        # Go to next state
        state = getState(observation)

env.close()