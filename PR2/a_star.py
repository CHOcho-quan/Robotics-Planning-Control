from numpy import loadtxt
import matplotlib.pyplot as plt

from robotplanner import heuristic
plt.ion()
import time
import numpy as np
import math
import time
import sys, multiprocessing
from queue import PriorityQueue
from multiprocessing.dummy import Pool

actions = [ 1, 0, -1, 0, 1, 1, -1, -1, 1]

class AstarNode:
    """
    Node for A* search
    
    """
    def __init__(self, x, y, g = 99999, h = 0, f = None, parent = None, opened = False, epsilon = 1.0):
        """
        Initialization function for JPS Nodes
        x - position x
        y - position y
        g / h / f - g-value / heuristic / g + h
        parent - parent node
        opened - if inserted to open list
        
        """
        # Coordinates
        self.x = x
        self.y = y

        # Costs / heuristics
        self.g = g
        self.h = h
        if f is None:
            self.f = self.g + self.h * epsilon
        else:
            self.f = f

        # Where are you from & in open list or not
        self.parent = parent
        self.opened = opened

    def __eq__(self, node):
        return (self.x == node.x) and (self.y == node.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __lt__(self, node):
        return self.f < node.f or ((self.f == node.f) and self.h < node.h)
    
    def moveCnt(self):
        return int(abs(self.x - self.parent.x) + abs(self.y - self.parent.y))

    def move(self):
        return (self.x - self.parent.x, self.y - self.parent.y)

class AStar:
    """
    A* Planner
    
    """
    def __init__(self, envmap, move_per_plan = 10):
        """
        Initiailization function
        
        """
        self._env = envmap
        self._gvalue = np.ones_like(envmap) * 999999
        self._close = set()
        self._open = PriorityQueue(maxsize=20000)
        self._parents = {}
        self._cnt = 1
        self._path = []
        self._nodes = {}
        self._per_plan = move_per_plan

    def heuristic(self, p):
        """
        L1 - Heuristic
        p - given point position
        
        """
        return abs(p[0] - self._goalx) + abs(p[1] - self._goaly)

    def plan(self, start, goal, n_expand = 10000):
        """
        Planning Core
        
        """
        self._cnt += 1
        if self._cnt % self._per_plan == 0:
            # Directly get path
            pass
        # A* Initialization
        self._gvalue = np.ones_like(self._env) * 999999
        self._close = set()
        self._nodes = {}
        self._open = PriorityQueue(maxsize=20000)
        self._goalx, self._goaly = goal
        newrobotpos = start
        self._gvalue[start] = 0
        close_queue = PriorityQueue(maxsize=20000)
        close_queue.put((self.heuristic(start), start))

        start_node = AstarNode(start[0], start[1], 0, self.heuristic(start))
        self._close.add(start_node)
        goal_node = AstarNode(goal[0], goal[1])
        start_node.opened = True
        self._nodes[start] = start_node
        self._open.put(start_node)
        self._path = []
        self._parents = {}

        cnt = 0
        while (goal_node not in self._close):
            if self._open.empty():
                print("EMPTY OPEN")
                break
            cur_node = self._open.get()
            self._close.add(cur_node)
            cur_pos = np.array([cur_node.x, cur_node.y])
            cur_pos_tup = tuple(cur_pos)

            for i in range(len(actions)):
                action = np.array(actions[i : i + 2])
                next_pos = cur_pos + action
                next_pos_tup = tuple(next_pos)
                # closed move
                if next_pos_tup in self._nodes.keys() and self._nodes[next_pos_tup] in self._close:
                    continue
                # Invalid move
                if next_pos[0] < 0 or next_pos[0] >= self._env.shape[0] or \
                    next_pos[1] < 0 or next_pos[1] >= self._env.shape[1] or self._env[next_pos_tup] == 1:
                    continue
                if self._gvalue[next_pos_tup] > self._gvalue[cur_pos_tup] + 1:
                    self._parents[next_pos_tup] = cur_pos_tup
                    self._gvalue[next_pos_tup] = self._gvalue[cur_pos_tup] + 1
                    if next_pos_tup in self._nodes.keys() and self._nodes[next_pos_tup].opened:
                        self._nodes[next_pos_tup].g = self._gvalue[cur_pos_tup] + 1
                        self._nodes[next_pos_tup].parent = cur_node
                    else:
                        n_node = AstarNode(next_pos[0], next_pos[1], self._gvalue[next_pos_tup], self.heuristic(next_pos_tup), parent=cur_node, opened=True)
                        self._nodes[next_pos_tup] = n_node
                        self._open.put(n_node)
                
                cnt += 1

        # Get best action
        cur_pos_tup = goal
        nxt = cur_pos_tup
        while nxt in self._parents.keys():
            self._path.append(nxt)
            nxt = self._parents[nxt]

        # print(self._path)
        if len(self._path) > 1:
            newrobotpos = self._path[-1]

        return self._path, newrobotpos

if __name__ == "__main__":
    # robotstart = np.array([0, 2])
    # targetstart = np.array([5, 3])
    # envmap = loadtxt('./maps/map0.txt')
    # robotstart = np.array([699, 799])
    # targetstart = np.array([699, 1699])
    # envmap = loadtxt('./maps/map1.txt')
    # robotstart = np.array([0, 2])
    # targetstart = np.array([7, 9])
    # envmap = loadtxt('./maps/map2.txt')
    # robotstart = np.array([249, 249])
    # targetstart = np.array([399, 399])
    # envmap = loadtxt('./maps/map3.txt')
    # robotstart = np.array([0, 0])
    # targetstart = np.array([5, 6])
    # envmap = loadtxt('./maps/map4.txt')
    # robotstart = np.array([0, 0])
    # targetstart = np.array([29, 59])
    # envmap = loadtxt('./maps/map5.txt')
    # robotstart = np.array([0, 0])
    # targetstart = np.array([29, 36])
    # envmap = loadtxt('./maps/map6.txt')

    robotstart = np.array([249, 1199])
    targetstart = np.array([1649, 1899])
    envmap = loadtxt('./maps/map1.txt')
    # robotstart = np.array([74, 249])
    # targetstart = np.array([399, 399])
    # envmap = loadtxt('./maps/map3.txt')
    # robotstart = np.array([4, 399])
    # targetstart = np.array([399, 399])
    # envmap = loadtxt('./maps/map3.txt')

    astar = AStar(envmap)
    ts = time.time()
    path, _ = astar.plan(tuple(robotstart), tuple(targetstart))
    print('%s took: %s sec.\n' % ("it",(time.time() - ts)))
    f, ax = plt.subplots()
    ax.imshow( envmap.T, interpolation="none", cmap='gray_r', origin='lower', \
                extent=(-0.5, envmap.shape[0]-0.5, -0.5, envmap.shape[1]-0.5) )
    ax.axis([-0.5, envmap.shape[0]-0.5, -0.5, envmap.shape[1]-0.5])
    ax.set_xlabel('x')
    ax.set_ylabel('y')  
    for node in path:
        hr = ax.plot(node[0], node[1], 'bs')
    plt.show()
    plt.savefig("f.png")
