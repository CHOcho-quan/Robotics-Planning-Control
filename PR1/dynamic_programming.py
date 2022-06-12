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

# Door Key State
UP_DN = 0 # Unpicked key door not open
PK_DN = 1 # Picked key door open
PK_DO = 2 # Picked key door unopen

# Direction
TOP = 0
LEFT = 1
DOWN = 2
RIGHT = 3

# Util
INF = 9999

class DynamicProgramming:
    """
    Dynamic Programming Algorithm Class
    Solves an input DP door key problem for a known map
    
    """
    def __init__(self, env, info):
        """
        Initialization function for the DP Class
        Input:
        env - inputting environment of gym
        info - dictionary of environment information
        
        """
        # Basic Information
        self._env = env
        self._info = info
        self._goal = info["goal_pos"]
        self._pos = info["init_agent_pos"]
        """
        UP: [0, -1], DOWN: [0, 1], RIGHT: [1, 0], LEFT: [-1, 0]
        """
        dir = info["init_agent_dir"]
        if np.array_equal(dir, np.array([0, -1])):
            self._dir = TOP
        elif np.array_equal(dir, np.array([0, 1])):
            self._dir = DOWN
        elif np.array_equal(dir, np.array([1, 0])):
            self._dir = RIGHT
        elif np.array_equal(dir, np.array([-1, 0])):
            self._dir = LEFT
        else: # should be impossible
            raise Exception("Unknown Direction Type {0}".format(dir))
        self._door = info["door_pos"]
        self._key = info["key_pos"]
        self._height = info["height"]
        self._width = info["width"]

        # Check if there's a valid path
        existPath = self._CheckPath()
        if not existPath:
            print("Not Exist Path from Start to End!")
            raise Exception("EnvError: There's no valid path from start to end")

        # DP Algorithm Settings
        self._action_space = [MF, TL, TR, PK, UD]
        self._doorkey_space = [UP_DN, PK_DN, PK_DO]
        self._direction = [TOP, LEFT, DOWN, RIGHT]
        self._dir_map = np.array([[0, -1], [-1, 0], [0, 1], [1, 0]])
        self._states, self._goal_inds = self._GenerateStates()
        self._state_num = len(self._states)
        self._cost = self._GenerateCostMatrix()

    def _isWall(self, i, j):
        """
        Check if a grid is wall
        Input:
        i, j - index of the grid
        Output:
        isWall - bool indicating if (i, j) grid is wall
        
        """
        cur_grid = self._env.grid.get(i, j)
        return cur_grid is not None and cur_grid.type == 'wall'

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
                    for l in self._doorkey_space:
                        if self._isWall(i, j):
                            continue
                        cur_state = {
                            "pos": np.array([i, j]),
                            "keyState": l,
                            "dir": k
                        }
                        self._state_dict[(i, j, l, k)] = len(states)
                        if i == self._goal[0] and j == self._goal[1]:
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
            if np.array_equal(cur_state["pos"], self._door) and cur_state["keyState"] != 2:
                continue
            for act in self._action_space:
                if act == MF:
                    n_pos = cur_state["pos"] + self._dir_map[cur_state["dir"]]
                    if self._isWall(n_pos[0], n_pos[1]):
                        continue
                    if np.array_equal(n_pos, self._door) and cur_state["keyState"] != PK_DO:
                        continue
                    next_state_id = self._state_dict[ \
                        (n_pos[0], n_pos[1], cur_state["keyState"], cur_state["dir"])]
                    cost_matrix[i, next_state_id] = 1
                elif act == TL:
                    next_state_id = self._state_dict[ \
                        (cur_state["pos"][0], cur_state["pos"][1], cur_state["keyState"], \
                            (cur_state["dir"] + 1) % 4)]
                    cost_matrix[i, next_state_id] = 1
                elif act == TR:
                    next_state_id = self._state_dict[ \
                        (cur_state["pos"][0], cur_state["pos"][1], cur_state["keyState"], \
                            (cur_state["dir"] + 3) % 4)]
                    cost_matrix[i, next_state_id] = 1
                elif act == PK:
                    if np.array_equal(cur_state["pos"] + self._dir_map[cur_state["dir"]], self._key) \
                        and cur_state["keyState"] == UP_DN:
                        next_state_id = self._state_dict[ \
                            (cur_state["pos"][0], cur_state["pos"][1], cur_state["keyState"] + 1, \
                                cur_state["dir"])]
                        cost_matrix[i, next_state_id] = 1
                elif act == UD:
                    if np.array_equal(cur_state["pos"] + self._dir_map[cur_state["dir"]], self._door) \
                        and cur_state["keyState"] == PK_DN:
                        next_state_id = self._state_dict[ \
                            (cur_state["pos"][0], cur_state["pos"][1], cur_state["keyState"] + 1, \
                                cur_state["dir"])]
                        cost_matrix[i, next_state_id] = 1
                else: # should be impossible
                    raise Exception("Unknow action {0}".format(act))

        return cost_matrix

    def _CheckPath(self):
        """
        BFS to check if there's a valid path from start to end
        Output:
        success - True for existing valid path vice versa
        
        """
        q = queue.Queue()
        q.put(self._pos)

        # Supports
        action = [1, 0, -1, 0, 1]
        visited = np.zeros((self._width, self._height), dtype = np.bool)
        visited[self._pos[0], self._pos[1]] = 1
        can_access_key = False

        while not q.empty():
            cur_pos = q.get()
            if self._env.grid.get(cur_pos[0], cur_pos[1]) is not None and \
                self._env.grid.get(cur_pos[0], cur_pos[1]).type == 'door':
                # At door
                if not can_access_key:
                    if q.empty():
                        return False
                    q.put(cur_pos)
                    continue

            for i in range(4):
                n_pos = cur_pos + np.array(action[i:i + 2])
                # Invalid places
                if visited[n_pos[0], n_pos[1]] or n_pos[0] < 0 or n_pos[0] >= self._width \
                    or n_pos[1] < 0 or n_pos[1] >= self._height:
                    continue
                cur_grid = self._env.grid.get(n_pos[0], n_pos[1])
                # Normal Cell
                if cur_grid is None:
                    q.put(n_pos)
                    visited[n_pos[0], n_pos[1]] = 1
                    continue
                # Walls
                if cur_grid.type == 'wall':
                    continue
                # Key
                if cur_grid.type == 'key':
                    can_access_key = True
                    q.put(n_pos)
                    visited[n_pos[0], n_pos[1]] = 1
                # Door
                if cur_grid.type == 'door':
                    q.put(n_pos)
                    visited[n_pos[0], n_pos[1]] = 1
                # Goal
                if cur_grid.type == 'goal':
                    return True

        return False

    def _GetPolicy(self, policy, debug=False):
        """
        Get Action sequence from generated policy
        Output:
        actions - sequence of actions that minimizes costs
        
        """
        start_id = self._state_dict[(self._pos[0], self._pos[1], UP_DN, self._dir)]
        states_trans = [start_id]
        for i in range(len(policy) - 1, -1, -1):
            n_state = policy[i][start_id]
            if start_id == n_state:
                continue
            start_id = n_state
            states_trans.append(n_state)
        # Debug print
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
                        and state1["keyState"] == state2["keyState"]:
                        actions.append(act)
                        break
                elif act == TL:
                    if np.array_equal(state1["pos"], state2["pos"]) and \
                        state1["keyState"] == state2["keyState"] and\
                        state2["dir"] == (state1["dir"] + 1) % 4:
                        actions.append(act)
                        break
                elif act == TR:
                    if np.array_equal(state1["pos"], state2["pos"]) and \
                        state1["keyState"] == state2["keyState"] and\
                        state2["dir"] == (state1["dir"] + 3) % 4:
                        actions.append(act)
                        break
                elif act == PK:
                    if np.array_equal(state1["pos"], state2["pos"]) and \
                        state1["dir"] == state2["dir"] and state1["keyState"] == UP_DN \
                        and state2["keyState"] == PK_DN:
                        actions.append(act)
                        break
                elif act == UD:
                    if np.array_equal(state1["pos"], state2["pos"]) and \
                        state1["dir"] == state2["dir"] and state1["keyState"] == PK_DN \
                        and state2["keyState"] == PK_DO:
                        actions.append(act)
                        break
                else: # should not be possible
                    raise Exception("Unknown action ID: {0}".format(act))
        
        return actions

    def Solve(self, debug=False):
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

        actions = self._GetPolicy(policy, debug)
        return actions

# Simple test
if __name__ == '__main__':
    env, info = load_env('./envs/doorkey-8x8-normal.env')
    dp = DynamicProgramming(env, info)
    act_seq = dp.Solve(True)
    print(len(act_seq))
    draw_process_from_seq(act_seq, 6, 4, env)
