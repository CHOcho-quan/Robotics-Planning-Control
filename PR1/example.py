import numpy as np
import gym
from utils import *

MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

def example_use_of_gym_env():
    '''
    The Coordinate System:
        (0,0): Top Left Corner
        (x,y): x-th column and y-th row
    '''
    
    print('<========== Example Usages ===========> ')
    env_path = './envs/example-8x8.env'
    # env, info = load_env(env_path) # load an environment
    
    env, info = load_env(env_path)
    print('<Environment Info>\n')
    print(info) # Map size
                # agent initial position & direction, 
                # key position, door position, goal position
    print('<================>\n')
    for i in range(info['height']):
        for j in range(info['width']):
            if (env.grid.get(i, j) != None):
                print(env.grid.get(i, j).type)
    
    # Visualize the environment
    plot_env(env) 
    
    
    # Get the agent position
    agent_pos = env.agent_pos
    
    # Get the agent direction
    agent_dir = env.dir_vec # or env.agent_dir
    
    # Get the cell in front of the agent
    front_cell = env.front_pos # == agent_pos + agent_dir
    
    # Access the cell at coord: (2,3)
    cell = env.grid.get(2, 3) # NoneType, Wall, Key, Goal
    
    # Get the door status
    door = env.grid.get(info['door_pos'][0], info['door_pos'][1])
    is_open = door.is_open
    is_locked = door.is_locked
    
    # Determine whether agent is carrying a key
    is_carrying = env.carrying is not None
    
    # Take actions
    cost, done = step(env, TL) # MF=0, TL=1, TR=2, PK=3, UD=4
    print('Moving Forward Costs: {}'.format(cost))
    print(env.agent_pos, env.dir_vec)
    plot_env(env) 
    cost, done = step(env, TL) # MF=0, TL=1, TR=2, PK=3, UD=4
    print('Turning Left Costs: {}'.format(cost))
    print(env.agent_pos, env.dir_vec)
    plot_env(env) 
    cost, done = step(env, TL) # MF=0, TL=1, TR=2, PK=3, UD=4
    print(env.agent_pos, env.dir_vec)
    plot_env(env) 
    print('Turning Right Costs: {}'.format(cost))
    cost, done = step(env, PK) # MF=0, TL=1, TR=2, PK=3, UD=4
    print('Picking Up Key Costs: {}'.format(cost))
    cost, done = step(env, UD) # MF=0, TL=1, TR=2, PK=3, UD=4
    print('Unlocking Door Costs: {}'.format(cost))   
    
    # Determine whether we stepped into the goal
    if done:
        print("Reached Goal")
    
    # The number of steps so far
    print('Step Count: {}'.format(env.step_count))
    
if __name__ == '__main__':
    example_use_of_gym_env()
