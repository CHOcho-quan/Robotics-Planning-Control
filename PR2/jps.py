import imageio
from numpy import loadtxt
import matplotlib.pyplot as plt
plt.ion()
import time
import numpy as np
import math
import time
import sys, multiprocessing
sys.setrecursionlimit(100000)
from queue import PriorityQueue
from multiprocessing.dummy import Pool

class JPSNode:
    """
    Node for jump point search
    
    """
    def __init__(self, x, y, dx, dy, g = 99999, h = 0, f = None, parent = None, opened = False, epsilon = 1.0):
        """
        Initialization function for JPS Nodes
        x - position x
        y - position y
        dx - move of coming to this node x
        dy - move of coming to this node y
        g / h / f - g-value / heuristic / g + h
        parent - parent node
        opened - if inserted to open list
        
        """
        # Coordinates
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy

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

class JPS:
    """
    Implementation of JPS algorithm
    
    """
    def __init__(self, envmap):
        """
        Initialization function
        envmap - environment
        
        """
        self._env = envmap
        self._open = PriorityQueue(maxsize=20000)
        self._nodes = {}
        self._path = []
        self._close = set()
        self._jmp = {}

        # Neighbour setting for move 0, 1, 2
        self._neighbor_num = [[8, 0], [1, 2], [3, 2]]
        # Natural neighbor indexed by move type / move id
        self._natural_neighbor = {(0, 0): [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]],
                                  (1, 0): [[1, 0]], (0, 1): [[0, 1]], (0, -1): [[0, -1]], (-1, 0): [[-1, 0]],
                                  (1, 1): [[1, 0], [0, 1], [1, 1]], (1, -1): [[1, 0], [0, -1], [1, -1]],
                                  (-1, 1): [[-1, 0], [0, 1], [-1, 1]], (-1, -1): [[-1, 0], [0, -1], [-1, -1]]}
        self._forced_check = {(1, 0): [[0, 1], [0, -1]], (-1, 0): [[0, 1], [0, -1]], (0, 1): [[1, 0], [-1, 0]], (0, -1): [[1, 0], [-1, 0]],
                              (1, 1): [[-1, 0], [0, -1]], (1, -1): [[-1, 0], [0, 1]], (-1, 1): [[1, 0], [0, -1]], (-1, -1): [[1, 0], [0, 1]]}
        self._forced_neighbor = {(1, 0): [[1, 1], [1, -1]], (-1, 0): [[-1, 1], [-1, -1]], (0, 1): [[1, 1], [-1, 1]], (0, -1): [[1, -1], [-1, -1]],
                                 (1, 1): [[-1, 1], [1, -1]], (1, -1): [[-1, -1], [1, 1]], (-1, 1): [[1, 1], [-1, -1]], (-1, -1): [[-1, 1], [1, -1]]}

    def heuristic_l1(self, p):
        """
        L1 - Heuristic
        p - given point position
        
        """
        return abs(p[0] - self._goalx) + abs(p[1] - self._goaly)

    def isFree(self, x, y):
        """
        Check if a node is free
        x, y - position of a node
        
        """
        return x >= 0 and x < self._env.shape[0] and \
               y >= 0 and y < self._env.shape[1] and self._env[x, y] == 0

    def _getPath(self, e, s):
        """
        Get path from JPS
        e / s - target / start point position tuple
        
        """
        node = self._nodes[e]
        self._path.append(node)
        while (node and node.x != s[0] or node.y != s[1]):
            node = self._nodes[(node.parent.x, node.parent.y)]
            self._path.append(node)

        return self._path

    def getPath(self):
        return self._path

    def jump(self, x, y, dx, dy):
        """
        Jump function to neighbors, check if forced neighbors exist
        x, y  - position
        dx, dy - how did you come here
        
        """
        if (x, y, dx, dy) in self._jmp.keys():
            return self._jmp[(x, y, dx, dy)]
        nx = x + dx
        ny = y + dy
        if not self.isFree(nx, ny):
            return False, nx, ny

        # For goal
        if self._goalx == nx and self._goaly == ny:
            return True, nx, ny

        # Check force
        for i in range(2):
            mx = nx + self._forced_check[(dx, dy)][i][0]
            my = ny + self._forced_check[(dx, dy)][i][1]
            if not self.isFree(mx, my):
                return True, nx, ny

        moveCnt = abs(dx) + abs(dy)
        for i in range(self._neighbor_num[moveCnt][0] - 1):
            forced, nnx, nny = self.jump(nx, ny, self._natural_neighbor[(dx, dy)][i][0], self._natural_neighbor[(dx, dy)][i][1])
            if forced:
                return True, nx, ny
        forced, nx, ny = self.jump(nx, ny, dx, dy)
        return forced, nx, ny

    def new_jump(self, x, y, dx, dy):
        """
        Jump without recursion
        x, y  - position
        dx, dy - how did you come here
        
        """
        stack = [(x, y, dx, dy, -1)]
        ret_register = None
        while True:
            # print(len(stack), ret_register)
            # no level in stack
            if len(stack) == 0:
                return ret_register

            # print(stack[-1])
            # get stack top
            x, y, dx, dy, ret_pos = stack[-1]

            nx = x + dx
            ny = y + dy
            if (ret_pos == -1):
                if not self.isFree(nx, ny):
                    ret_register = False, nx, ny
                    stack.pop()
                    continue
                # For goal
                if self._goalx == nx and self._goaly == ny:
                    ret_register = True, nx, ny
                    stack.pop()
                    continue

                # Check force
                continue_flag = False
                for i in range(2):
                    mx = nx + self._forced_check[(dx, dy)][i][0]
                    my = ny + self._forced_check[(dx, dy)][i][1]
                    if not self.isFree(mx, my):
                        ret_register = True, nx, ny
                        stack.pop()
                        continue_flag = True
                        break
                if continue_flag:
                    continue
            if (ret_pos < 999):
                moveCnt = abs(dx) + abs(dy)
                continue_flag = False
                for i in range(self._neighbor_num[moveCnt][0] - 1):
                    if (ret_pos > i):
                        continue
                    if (ret_pos == i):
                        forced, nnx, nny = ret_register
                        if forced:
                            ret_register = True, nx, ny
                            stack.pop()
                            continue_flag = True
                            break
                        continue
                    stack[-1] = x, y, dx, dy, i
                    stack.append((nx, ny, self._natural_neighbor[(dx, dy)][i][0], self._natural_neighbor[(dx, dy)][i][1], -1))
                    continue_flag = True
                    break
                if (continue_flag):
                    continue
                stack[-1] = x, y, dx, dy, 999
                stack.append((nx, ny, dx, dy, -1))
            if (ret_pos == 999):
                stack.pop()
                assert(ret_register != None)
                continue

    def getSucc(self, node, heuristic):
        """
        Get all successors according to JPS rule
        node - current node to get successors
        heuristic - utilized heuristic function
        
        """
        succs = []
        succs_cost = []
        x, y = node.x, node.y

        moveCnt = abs(node.dx) + abs(node.dy)
        move = (node.dx, node.dy)
        natural_num, forced_num = self._neighbor_num[moveCnt]
        for i in range(natural_num + forced_num):
            if i < natural_num:
                # Deal with natural neighbours
                dx, dy = self._natural_neighbor[move][i]
                has_forced, nx, ny = self.jump(x, y, dx, dy)
                self._jmp[(x, y, dx, dy)] = [has_forced, nx, ny]
                if not has_forced:
                    continue
            else:
                # Forced Neighbours
                ind = i - natural_num
                nx = x + self._forced_check[move][ind][0]
                ny = y + self._forced_check[move][ind][1]
                if not self.isFree(nx, ny):
                    dx = self._forced_neighbor[(dx, dy)][ind][0]
                    dy = self._forced_neighbor[(dx, dy)][ind][1]
                    has_forced, nx, ny = self.jump(x, y, dx, dy)
                    self._jmp[(x, y, dx, dy)] = [has_forced, nx, ny]
                    if not has_forced:
                        continue
                else:
                    continue

            if (nx, ny) not in self._nodes.keys():
                self._nodes[(nx, ny)] = JPSNode(nx, ny, dx, dy)
                self._nodes[(nx, ny)].h = heuristic((nx, ny))

            succs.append((nx, ny))
            succs_cost.append(math.sqrt((nx - x) ** 2) + math.sqrt((ny - y) ** 2))

        return succs, succs_cost

    def updateSuccs(self, succ, succ_cost):
        # succ, succ_cost = succ_zip
        sx, sy = succ
        child = self._nodes[(sx, sy)]
        cur_g = self._curnode.g + succ_cost

        if cur_g < child.g:
            child.g = cur_g
            child.parent = self._curnode

            if not child.opened:
                child.opened = True
                self._open.put(child)
            else:
                child.dx = child.x - self._curnode.x
                child.dy = child.y - self._curnode.y
                if child.dx != 0:
                    child.dx = int(child.dx / abs(child.dx))
                if child.dy != 0:
                    child.dy = int(child.dy / abs(child.dy))

    def plan(self, start, goal, heuristic="l1", maxExpand=100):
        """
        JPS Planning algorithm
        start / goal - start / goal position np array
        heuristic - which heuristic to use
        
        """
        self._open = PriorityQueue(maxsize=20000)
        self._nodes = {}
        self._path = []
        self._close = set()
        self._jmp = {}
        heuristic_func = None
        if heuristic == "l1":
            heuristic_func = self.heuristic_l1

        goal_node = JPSNode(goal[0], goal[1], 0, 0)
        self._goalx = goal[0]
        self._goaly = goal[1]
        start_node = JPSNode(start[0], start[1], 0, 0, 0, heuristic_func(start))
        start_node.opened = True
        self._open.put(start_node)
        self._nodes[(start[0], start[1])] = start_node

        expandCnt = 0
        while goal_node not in self._close:
            expandCnt += 1
            if self._open.empty():
                print("OPEN EMPTY")
                break
            cur_node = self._open.get()
            self._close.add(cur_node)

            succs, succ_costs = self.getSucc(cur_node, heuristic_func)

            # Parallel Computing
            # self._curnode = cur_node
            # pool = Pool(multiprocessing.cpu_count())
            # pool.starmap(self.updateSuccs, zip(succs, succ_costs))
            for i, succ in enumerate(succs):
                sx, sy = succ
                child = self._nodes[(sx, sy)]
                cur_g = cur_node.g + succ_costs[i]

                if cur_g < child.g:
                    child.g = cur_g
                    child.parent = cur_node

                    if not child.opened:
                        child.opened = True
                        self._open.put(child)
                    else:
                        child.dx = child.x - cur_node.x
                        child.dy = child.y - cur_node.y
                        if child.dx != 0:
                            child.dx = int(child.dx / abs(child.dx))
                        if child.dy != 0:
                            child.dy = int(child.dy / abs(child.dy))

        self._path = self._getPath(goal, start)
        newrobotpos = np.array(start)
        for di, dj in self._natural_neighbor[(0, 0)]:
            if self._jmp[(start[0], start[1], di, dj)][1] == self._path[-2].x and \
                self._jmp[(start[0], start[1], di, dj)][2] == self._path[-2].y:
                newrobotpos[0] += di
                newrobotpos[1] += dj

        return self._path, newrobotpos

if __name__ == "__main__":
    robotstart = np.array([249, 1199])
    targetstart = np.array([1649, 1899])
    envmap = loadtxt('./maps/map1.txt')
    jps = JPS(envmap)
    ts = time.time()
    path, _ = jps.plan(tuple(robotstart), tuple(targetstart))
    print('%s took: %s sec.\n' % ("it",(time.time() - ts)))
    f, ax = plt.subplots()
    ax.imshow( envmap.T, interpolation="none", cmap='gray_r', origin='lower', \
                extent=(-0.5, envmap.shape[0]-0.5, -0.5, envmap.shape[1]-0.5) )
    ax.axis([-0.5, envmap.shape[0]-0.5, -0.5, envmap.shape[1]-0.5])
    ax.set_xlabel('x')
    ax.set_ylabel('y')  
    for node in path:
        hr = ax.plot(node.x, node.y, 'bs')
    plt.show()
    plt.savefig("f.png")
