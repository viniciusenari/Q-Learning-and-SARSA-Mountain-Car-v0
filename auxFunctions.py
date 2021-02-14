import numpy as np
import gym
import pickle

env = gym.make('MountainCar-v0')

# Discritize observation and action space in bins.
pos_space = np.linspace(-1.2, 0.6, 18)
vel_space = np.linspace(-0.07, 0.07, 28)

# given observation, returns what bin
def getState(observation):
    pos, vel = observation
    pos_bin = np.digitize(pos, pos_space)
    vel_bin = np.digitize(vel, vel_space)
 
    return (pos_bin, vel_bin)

# Creates a new empty Q-table for this environment
def createEmptyQTable():
    states = []
    for pos in range(len(pos_space) + 1):
        for vel in range(len(vel_space) + 1):
            states.append((pos,vel))
        
    Q = {}
    for state in states:
        for action in range(env.action_space.n):
            Q[state, action] = 0
    return Q

# Given a state and a set of actions
# returns action that has the highest Q-value
def maxAction(Q, state, actions=[0, 1, 2]):
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)
    return action

# Saves a variable as a file
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# Load a variable from file
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)