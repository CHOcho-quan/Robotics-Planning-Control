import os
import numpy as np
import gym
import gym_minigrid
import pickle
import imageio
import matplotlib.pyplot as plt
import queue

from utils import *

# Moving Action
MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

# Key State
UP = 0 # Unpicked key
PKED = 1 # Picked key

# Door State
OO = 0 # Two door closed
CO = 1 # First open second closed
OC = 2 # First closed second open
CC = 3 # Both open

# Goal Pos State
G1 = 0 # Goal at (5, 1)
G2 = 1 # Goal at (6, 3)
G3 = 2 # Goal at (5, 6)

# Key Pos State
K1 = 0 # Key at (1, 1)
K2 = 1 # Key at (2, 3)
K3 = 2 # Key at (1, 6)

# Direction
TOP = 0
LEFT = 1
DOWN = 2
RIGHT = 3

# Util
INF = 9999

class DynamicProgrammingRD:
    """
    Dynamic Programming Algorithm Class
    Solves an input DP door key problem for a random map

    """
    def __init__(self):
        """
        Initialization function for the DP Class
        Input:
        env - inputting environment of gym
        info - dictionary of environment information

        """
        # Deterministic Information
        self._height = 8
        self._width = 8

        # DP Algorithm Settings
        self._policy = None
        # State Space
        self._goal_space = [G1, G2, G3]
        self._goal_map = np.array([[5, 1], [6, 3], [5, 6]])
        self._door_space = [OO, CO, OC, CC]
        self._key_space = [UP, PKED]
        self._keypos = [K1, K2, K3]
        self._key_map = np.array([[1, 1], [2, 3], [1, 6]])
        self._direction = [TOP, LEFT, DOWN, RIGHT]
        # Action Space
        self._action_space = [MF, TL, TR, PK, UD]
        self._dir_map = np.array([[0, -1], [-1, 0], [0, 1], [1, 0]])
        self._states, self._goal_inds = self._GenerateStates()
        self._state_num = len(self._states)
        self._cost = self._GenerateCostMatrix()

    def _isDoor(self, i, j):
        """
        Check if a grid is door
        Input:
        i, j - index of the grid
        Output:
        door_num - No.1 door or No.2 door, negative means not a door
        
        """
        if i != 4:
            return -1
        if j == 2:
            # Door No.1
            return 1
        elif j == 5:
            return 2
        else:
            return -1

    def _isWall(self, i, j):
        """
        Check if a grid is wall
        Input:
        i, j - index of the grid
        Output:
        isWall - bool indicating if (i, j) grid is wall
        
        """
        # Wall at perimeter of the puzzle
        if i == 0 or j == 0 or i == self._width - 1 or j == self._height - 1:
            return True
        if i != 4:
            return False
        else:
            # Door at (4, 2) and (4, 5)
            return j != 2 and j != 5

    def _GenerateStates(self):
        """
        Generate states for later usage
        Output:
        states - list of states

        """
        goals = []
        states = []
        self._state_dict = {}
        for i in range(self._width):
            for j in range(self._height):
                for k in self._direction:
                    for l in self._goal_space:
                        for m in self._keypos:
                            for n in self._door_space:
                                for o in self._key_space:
                                    if self._isWall(i, j):
                                        continue
                                    cur_state = {
                                        "pos": np.array([i, j]),
                                        "dir": k,
                                        "goal_pos": l,
                                        "key_pos": m,
                                        "door": n,
                                        "key": o
                                    }
                                    self._state_dict[(i, j, k, l, m, n, o)] = len(states)
                                    if l == G1 and i == 5 and j == 1:
                                        goals.append(len(states))
                                    elif l == G2 and i == 6 and j == 3:
                                        goals.append(len(states))
                                    elif l == G3 and i == 5 and j == 6:
                                        goals.append(len(states))
                                    states.append(cur_state)

        return states, goals

    def _GenerateCostMatrix(self):
        """
        Generate a cost matrix at each state w.r.t next state
        Output:
        cost_matrix - a matrix containing costs of at each state

        """
        cost_matrix = np.ones((self._state_num, self._state_num)) * INF
        for i in range(self._state_num):
            cost_matrix[i, i] = 0
            cur_state = self._states[i]
            # At door while the door is not open, you can do nothing
            door_num = self._isDoor(cur_state["pos"][0], cur_state["pos"][1])
            if door_num == 1 and cur_state["door"] != CC and cur_state["door"] != CO:
                continue
            if door_num == 2 and cur_state["door"] != CC and cur_state["door"] != OC:
                continue
            for act in self._action_space:
                if act == MF:
                    n_pos = cur_state["pos"] + self._dir_map[cur_state["dir"]]
                    # Wall case
                    if self._isWall(n_pos[0], n_pos[1]):
                        continue
                    # Closed Door Case
                    n_door = self._isDoor(n_pos[0], n_pos[1])
                    if n_door == 1 and cur_state["door"] != CC and cur_state["door"] != CO:
                        continue
                    if n_door == 2 and cur_state["door"] != CC and cur_state["door"] != OC:
                        continue
                    next_state_id = self._state_dict[ \
                        (n_pos[0], n_pos[1], cur_state["dir"], cur_state["goal_pos"], \
                            cur_state["key_pos"], cur_state["door"], cur_state["key"])]
                    cost_matrix[i, next_state_id] = 1
                elif act == TL:
                    next_state_id = self._state_dict[ \
                        (cur_state["pos"][0], cur_state["pos"][1], (cur_state["dir"] + 1) % 4, \
                            cur_state["goal_pos"], cur_state["key_pos"], cur_state["door"], cur_state["key"])]
                    cost_matrix[i, next_state_id] = 1
                elif act == TR:
                    next_state_id = self._state_dict[ \
                        (cur_state["pos"][0], cur_state["pos"][1], (cur_state["dir"] + 3) % 4, \
                            cur_state["goal_pos"], cur_state["key_pos"], cur_state["door"], cur_state["key"])]
                    cost_matrix[i, next_state_id] = 1
                elif act == PK:
                    if cur_state["key"] == PKED:
                        continue
                    n_pos = cur_state["pos"] + self._dir_map[cur_state["dir"]]
                    if np.array_equal(n_pos, self._key_map[cur_state["key_pos"]]) and cur_state["key"] == UP:
                        next_state_id = self._state_dict[ \
                            (cur_state["pos"][0], cur_state["pos"][1], cur_state["dir"], cur_state["goal_pos"], \
                                cur_state["key_pos"], cur_state["door"], cur_state["key"] + 1)]
                        cost_matrix[i, next_state_id] = 1
                elif act == UD:
                    n_pos = cur_state["pos"] + self._dir_map[cur_state["dir"]]
                    n_door = self._isDoor(n_pos[0], n_pos[1])
                    if n_door < 0:
                        continue
                    if cur_state["key"] == PKED and cur_state["door"] + n_door < 4:
                        next_state_id = self._state_dict[ \
                                (cur_state["pos"][0], cur_state["pos"][1], cur_state["dir"], cur_state["goal_pos"], \
                                    cur_state["key_pos"], cur_state["door"] + n_door, cur_state["key"])]
                        cost_matrix[i, next_state_id] = 1
                else: # should be impossible
                    raise Exception("Unknow action {0}".format(act))

        return cost_matrix

    def GetPolicy(self, init_x, debug=False):
        """
        Get Action sequence from generated policy
        Input:
        init_x - seven-element tuple representing initial state of the robot
        Output:
        actions - sequence of actions that minimizes costs
        
        """
        policy = self._policy
        start_id = self._state_dict[init_x]
        states_trans = [start_id]
        for i in range(len(policy) - 1, -1, -1):
            n_state = policy[i][start_id]
            if start_id == n_state:
                continue
            start_id = n_state
            states_trans.append(n_state)
        # Debug output
        if debug:
            for s in states_trans:
                print(self._states[s])
        actions = []
        for s1, s2 in zip(states_trans, states_trans[1:]):
            state1 = self._states[s1]
            state2 = self._states[s2]
            # Here assume neighboring state is appliable
            for act in self._action_space:
                if act == MF:
                    n_pos = state1["pos"] + self._dir_map[state1["dir"]]
                    if state1["dir"] == state2["dir"] and np.array_equal(n_pos, state2["pos"]) \
                        and state1["key"] == state2["key"]:
                        actions.append(act)
                        break
                elif act == TL:
                    if np.array_equal(state1["pos"], state2["pos"]) and \
                        state1["key"] == state2["key"] and \
                        state2["dir"] == (state1["dir"] + 1) % 4:
                        actions.append(act)
                        break
                elif act == TR:
                    if np.array_equal(state1["pos"], state2["pos"]) and \
                        state1["key"] == state2["key"] and\
                        state2["dir"] == (state1["dir"] + 3) % 4:
                        actions.append(act)
                        break
                elif act == PK:
                    if np.array_equal(state1["pos"], state2["pos"]) and \
                        state1["dir"] == state2["dir"] and state1["key"] == UP \
                        and state2["key"] == PKED:
                        actions.append(act)
                        break
                elif act == UD:
                    n_pos = state1["pos"] + self._dir_map[state1["dir"]]
                    n_door = self._isDoor(n_pos[0], n_pos[1])
                    if n_door < 0:
                        continue
                    if np.array_equal(state1["pos"], state2["pos"]) and \
                        state1["key"] == PKED and state2["key"] == PKED and \
                        state1["dir"] == state2["dir"] and state1["door"] + n_door == state2["door"]:
                        actions.append(act)
                        break
                else: # should not be possible
                    raise Exception("Unknown action ID: {0}".format(act))
        
        return actions

    def Solve(self):
        """
        Solve the initialized problem
        Output:
        actions - sequence of actions that minimizes costs

        """
        # Basic Initialization
        state_size = self._state_num
        T = state_size - 1
        value_func = INF * np.ones((state_size, T))
        policy = []
        for i in self._goal_inds:
            value_func[i, -1] = 0

        # Backward iteration
        for j in range(T - 2, -1, -1):
            Q = self._cost + value_func[:, j + 1]
            policy.append(np.argmin(Q, axis=1))
            value_func[:, j] = np.amin(Q, axis=1)

            # Early Stopping by value function
            if np.array_equal(value_func[:, j], value_func[:, j + 1]):
                break

        self._policy = policy

# Simple test
if __name__ == '__main__':
    env, info, _ = load_specific_random_env("./envs/random_envs/DoorKey-8x8_14.pickle")
    # Direction
    dir = info["init_agent_dir"]
    if np.array_equal(dir, np.array([0, -1])):
        dir_m = TOP
    elif np.array_equal(dir, np.array([0, 1])):
        dir_m = DOWN
    elif np.array_equal(dir, np.array([1, 0])):
        dir_m = RIGHT
    elif np.array_equal(dir, np.array([-1, 0])):
        dir_m = LEFT
    else: # should be impossible
        raise Exception("Unknown Direction Type {0}".format(dir))
    # Goal Position
    goal = info["goal_pos"]
    if np.array_equal(goal, np.array([5, 1])):
        goal_m = G1
    elif np.array_equal(goal, np.array([6, 3])):
        goal_m = G2
    elif np.array_equal(goal, np.array([5, 6])):
        goal_m = G3
    else: # should be impossible
        raise Exception("Unknown Goal Position {0}".format(goal))
    # Key Position
    key = info["key_pos"]
    if np.array_equal(key, np.array([1, 1])):
        key_m = K1
    elif np.array_equal(key, np.array([2, 3])):
        key_m = K2
    elif np.array_equal(key, np.array([1, 6])):
        key_m = K3
    else: # should be impossible
        raise Exception("Unknown Key Position {0}".format(goal))
    # Door State
    door_state = OO
    doors_pos = info["door_pos"]
    doors_op = info["door_open"]
    for i in range(2):
        if np.array_equal(doors_pos[i], np.array([4, 2])) and doors_op[i]:
            door_state += 1
        elif np.array_equal(doors_pos[i], np.array([4, 5])) and doors_op[i]:
            door_state += 2

    init_x = (info["init_agent_pos"][0], info["init_agent_pos"][1], dir_m, goal_m, key_m, door_state, UP)
    dp = DynamicProgrammingRD()
    dp.Solve()
    act_seq = dp.GetPolicy(init_x, True)
    print(len(act_seq))
    # draw_gif_from_seq(act_seq, env, "./test_r.gif")
    draw_process_from_seq(act_seq, 3, 3, env)
