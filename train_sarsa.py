from auxFunctions import getState, createEmptyQTable, maxAction, save_obj
import gym
import random
import numpy as np

env = gym.make('MountainCar-v0')
env._max_episode_steps = 1000

# Create an empty Q-table
Q = createEmptyQTable()

# Hyperparameters 
alpha = 0.1 # Learning Rate
gamma = 0.9 # Discount Factor 
epsilon = 1 # e-Greedy 
episodes = 50000 # number of episodes

score = 0
# Variable to keep track of the total score obtained at each episode
total_score = np.zeros(episodes)

for i in range(episodes):
    if i % 500 == 0:
        print(f'episode: {i}, score: {score}, epsilon: {epsilon:0.3f}')
    
    observation = env.reset()
    state = getState(observation)

    # e-Greedy strategy
    # Explore random action with probability epsilon
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    # Take best action with probability 1-epsilon
    else:
        action = maxAction(Q, state)
    
    score = 0
    done = False
    while not done:
        # Take action and observe next state
        next_observation, reward, done, info = env.step(action)
        next_state = getState(next_observation)
        
        # Get next action following e-Greedy policy
        if random.uniform(0, 1) < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action= maxAction(Q, next_state)
        
        # Add reward to the score of the episode
        score += reward

        # Update Q value for state and action given the bellman equation
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action]) 
        
        # Move to next state, and next action
        state, action = next_state, next_action

    total_score[i] = score
    epsilon = epsilon - 2/episodes if epsilon > 0.01 else 0.01

# Save Q-table as .pkl file
save_obj(Q, 'pre-trained-SARSA')
