import numpy as np
import math, cv2
from numpy import loadtxt
from queue import PriorityQueue
from robotplanner import robotplanner_astar
from jps import JPSNode
import matplotlib.pyplot as plt

def runtest(mapfile, robotstart, targetstart):
    # current positions of the target and robot
    robotpos = np.copy(robotstart);
    targetpos = np.copy(targetstart);
  
    # environment
    envmap = loadtxt(mapfile)

    newrobotpos = robotplanner_astar(envmap, robotpos, targetpos)
    print(newrobotpos)

if __name__ == "__main__":
    # you should change the following line to test different maps
    robotstart = np.array([249, 1199])
    targetstart = np.array([1649, 1899])
    envmap = (1 - loadtxt('./maps/map3.txt')) * 255
    cv2.imwrite("obs2.png", np.transpose(envmap))
