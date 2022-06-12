import numpy as np
import math

def targetplanner(envmap, robotpos, targetpos, basepos, movetime):
  # all possible directions of the target
  numofdirs = 4
  dX = [-1, 0, 0, 1]
  dY = [ 0, -1, 1, 0]

  # all possible directions of the robot
  numofrobodirs = 8
  robot_dX = [-1, -1, -1, 0, 0, 1, 1, 1]
  robot_dY = [-1, 0, 1, -1, 1, -1, 0, 1]

  newtargetpos = np.copy(targetpos)
  
  for mind in range(movetime):
    end_pos_list = []
    start_pos = newtargetpos

    # all possible actions and next positions of the target
    for dd in range(numofdirs):
      newx = start_pos[0] + dX[dd]
      newy = start_pos[1] + dY[dd]
      
      if (newx >= 0 and newx < envmap.shape[0] and newy >= 0 and newy < envmap.shape[1]):
        if(envmap[newx, newy] == 0):
          end_pos_list.append(np.array([newx, newy]))
    
    min_dist_arr = []
    for end_pos in end_pos_list:
      min_dist = np.linalg.norm(end_pos - robotpos)

      # for each target position, compute the minimal distance from robot
      for robo_dd in range(numofrobodirs):
        robot_newx = robotpos[0] + robot_dX[robo_dd]
        robot_newy = robotpos[1] + robot_dY[robo_dd]
        if (robot_newx >= 0 and robot_newx < envmap.shape[0] and robot_newy >= 0 and robot_newy < envmap.shape[1]):
          if(envmap[robot_newx, robot_newy] == 0):
            dist = np.linalg.norm(end_pos - np.array([robot_newx, robot_newy]))
            if dist < min_dist:
              min_dist = dist
      min_dist_arr.append(min_dist)
    
    # choose the action that achieves the maximal minimal distance
    min_dist_arr = np.array(min_dist_arr)
    minimax_idx = np.argmax(min_dist_arr)
    newtargetpos = end_pos_list[minimax_idx]

  # print(newtargetpos)
  
  return newtargetpos

