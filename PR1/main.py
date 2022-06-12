import glob, pickle, argparse, time
from dynamic_programming import DynamicProgramming
from dp_random_map import *
from utils import *

parser = argparse.ArgumentParser(description="DP Arguments")
parser.add_argument('--part', type=str, default="A", help="Which part you wanna run")

def partA(env_path):
    env, info = load_env(env_path)
    dp = DynamicProgramming(env, info)
    act_seq = dp.Solve()
    draw_gif_from_seq(act_seq, env, "./gifs/{0}.gif".format(os.path.basename(env_path).split('.')[0]))

def partB(env_path, dp_rd):
    env, info, _ = load_specific_random_env(env_path)
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
    act_seq = dp_rd.GetPolicy(init_x)
    draw_gif_from_seq(act_seq, env, "./rd_gifs/{0}.gif".format(os.path.basename(env_path).split('.')[0]))

if __name__ == '__main__':
    args = parser.parse_args()
    st = time.time()
    if args.part == "A":
        envs = glob.glob("./envs/*.env")
        for env in envs:
            partA(env)
    else:
        envs = glob.glob("./envs/random_envs/*.pickle")
        # Generate general policy here
        dp_rd = DynamicProgrammingRD()
        dp_rd.Solve()
        for env in envs:
            partB(env, dp_rd)
    en = time.time()
    print("Total Time Consumption: {0} ms".format((en - st) * 1000))
