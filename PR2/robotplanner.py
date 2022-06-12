import numpy as np
import math
import time
from queue import PriorityQueue

actions = [ 1, 0, -1, 0, 1, 1, -1, -1, 1]

def heuristic(robotpos_tup, targetpos_tup):
  return abs(robotpos_tup[0] - targetpos_tup[0]) + abs(robotpos_tup[1] - targetpos_tup[1])

def robotplanner_greedy(envmap, robotpos, targetpos):
  """
  Greedy planner
  
  """
  numofdirs = 8
  dX = [-1, -1, -1, 0, 0, 1, 1, 1]
  dY = [-1,  0,  1, -1, 1, -1, 0, 1]
  
  # use the old position if we fail to find an acceptable move
  newrobotpos = np.copy(robotpos)

  # for now greedily move towards the target 
  # but this is the gateway function for your planner 
  mindisttotarget = 1000000
  for dd in range(numofdirs):
    newx = robotpos[0] + dX[dd]
    newy = robotpos[1] + dY[dd]
  
    if (newx >= 0 and newx < envmap.shape[0] and newy >= 0 and newy < envmap.shape[1]):
      if(envmap[newx, newy] == 0):
        disttotarget = math.sqrt((newx-targetpos[0])**2 + (newy-targetpos[1])**2)
        if(disttotarget < mindisttotarget):
          mindisttotarget = disttotarget
          newrobotpos[0] = newx
          newrobotpos[1] = newy
  return newrobotpos

def robotplanner_astar(envmap, robotpos, targetpos, n = 10000):
  """
  Astar planner
  
  """
  newrobotpos = np.copy(robotpos)

  # A* Initialization
  g_value = np.ones_like(envmap) * 999999
  start_pos_tup = tuple(robotpos)
  target_pos_tup = tuple(targetpos)
  g_value[start_pos_tup] = 0
  close = { start_pos_tup }
  close_queue = PriorityQueue(maxsize=20000)
  close_queue.put((heuristic(start_pos_tup, target_pos_tup), start_pos_tup))
  open = PriorityQueue(maxsize=20000)
  open.put((heuristic(start_pos_tup, target_pos_tup) + g_value[start_pos_tup], start_pos_tup))
  parents = {}

  # RTAA* Algorithm
  cnt = 0
  while (target_pos_tup not in close):
    if open.empty():
      break
    cur_node = open.get()
    close.add(cur_node[1])
    cur_pos = np.array(cur_node[1])
    cur_pos_tup = cur_node[1]
    close_queue.put((heuristic(cur_pos_tup, target_pos_tup), cur_pos_tup))
    if cnt == n:
      break

    for i in range(len(actions)):
      action = np.array(actions[i : i + 2])
      next_pos = cur_pos + action
      next_pos_tup = tuple(next_pos)
      # closed move
      if next_pos_tup in close:
        continue
      # Invalid move
      if next_pos[0] < 0 or next_pos[0] >= envmap.shape[0] or \
        next_pos[1] < 0 or next_pos[1] >= envmap.shape[1] or envmap[next_pos_tup] == 1:
        continue
      if g_value[next_pos_tup] > g_value[cur_node[1]] + 1:
        parents[next_pos_tup] = cur_node[1]
        g_value[next_pos_tup] = g_value[cur_node[1]] + 1
        open.put((heuristic(next_pos_tup, target_pos_tup) + g_value[next_pos_tup], next_pos_tup))
    
    cnt += 1

  # Get best action
  _, cur_pos_tup = close_queue.get()
  nxt = cur_pos_tup
  best_act = np.array([])
  while nxt in parents.keys():
    if parents[nxt] == start_pos_tup:
      best_act = nxt
    nxt = parents[nxt];

  if len(best_act) != 0:
    newrobotpos = best_act

  return newrobotpos
