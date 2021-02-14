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
# Variable to keep track of the total score obtained
# at each episode to plot it later
total_score = np.zeros(episodes)

for i in range(episodes):
    done = False
    observation = env.reset()
    state = getState(observation)
    
    if i % 500 == 0:
        print(f'episode: {i}, score: {score}, epsilon: {epsilon:0.3f}')
    
    score = 0
    while not done:
        # e-Greedy strategy
        # Explore random action with probability epsilon
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        # Take best action with probability 1-epsilon
        else:
            action = maxAction(Q, state)
        
        # Observe next state based on 
        next_observation, reward, done, info = env.step(action)
        next_state = getState(next_observation)
        
        # Add reward to the score of the episode
        score += reward
        
        # Get next action
        next_action = maxAction(Q, next_state)
        
        # Update Q value for state and action given the bellman equation
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action]) 
        
        # Move to next state
        state = next_state

    # Save score for this episode
    total_score[i] = score
    # Reduce epsilon 
    epsilon = epsilon - 2/episodes if epsilon > 0.01 else 0.01

# Save Q-table as .pkl file
save_obj(Q, 'Q-table-Q-Learning')
