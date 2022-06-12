import math
from time import time
import numpy as np
from utils import visualize
from casadi import *

# Simulation params
np.random.seed(10)
time_step = 0.5 # time between steps in seconds
sim_time = 120    # simulation time

# Car params
x_init = 1.5
y_init = 0.0
theta_init = np.pi/2
v_max = 1
v_min = 0
w_max = 1
w_min = -1

# This function returns the reference point at time step k
def lissajous(k):
    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2*np.pi/50
    b = 3*a
    T = np.round(2*np.pi/(a*time_step))
    k = k % T
    delta = np.pi/2
    xref = xref_start + A*np.sin(a*k*time_step + delta)
    yref = yref_start + B*np.sin(b*k*time_step)
    v = [A*a*np.cos(a*k*time_step + delta), B*b*np.cos(b*k*time_step)]
    thetaref = np.arctan2(v[1], v[0])
    return [xref, yref, thetaref]

# This function implements a simple P controller
def simple_controller(cur_state, ref_state):
    k_v = 0.55
    k_w = 1.0
    v = k_v*np.sqrt((cur_state[0] - ref_state[0])**2 + (cur_state[1] - ref_state[1])**2)
    v = np.clip(v, v_min, v_max)
    angle_diff = ref_state[2] - cur_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi ) - np.pi
    w = k_w*angle_diff
    w = np.clip(w, w_min, w_max)
    return [v,w]

# This function implements Certainty Equivalent Controller(CEC)
def cec_controller(cur_state, ref_states, T = 2):
    assert(len(ref_states) == T + 1)
    cur_state = np.array(cur_state)
    ref_states = np.array(ref_states)
    # Basic Parameters
    gamma = 0.9
    objective = 0
    var_list = []
    constraints = []
    constraint_lower = []
    constraint_upper = []
    initial_guess = []
    lbx = []
    ubx = []
    Q = SX.eye(2) * 5
    R = SX.eye(2)
    q = 100

    et = cur_state - ref_states[0]
    # First deal with t = 0
    vt = SX.sym("v0")
    wt = SX.sym("w0")
    var_list.append(vt)
    var_list.append(wt)
    initial_guess.append(0)
    initial_guess.append(0)
    lbx.append(0)
    ubx.append(1)
    lbx.append(-1)
    ubx.append(1)
    ut = vertcat(vt, wt)
    objective += ut.T @ R @ ut

    # For t > 0
    for i in range(1, T):
        # Environment Constraint
        et0 = et[0] + SX.cos(et[2] + ref_states[i][2]) * vt + ref_states[i-1][0] - ref_states[i][0]
        et1 = et[1] + SX.sin(et[2] + ref_states[i][2]) * vt + ref_states[i-1][1] - ref_states[i][1]
        pt = vertcat(et0, et1)
        et2 = et[2] + wt + ref_states[i-1][2] - ref_states[i][2]
        et = vertcat(et0, et1, et2)
        constraints.append((et0 + ref_states[i][0] + 2) ** 2 + (et1 + ref_states[i][1] + 2) ** 2)
        constraint_lower.append(0.6) # Intended, to keep the agent far away from obstacles
        constraint_upper.append(100000)

        constraints.append((et0 + ref_states[i][0] - 1) ** 2 + (et1 + ref_states[i][1] - 2) ** 2)
        constraint_lower.append(0.6) # Intended, to keep the agent far away from obstacles
        constraint_upper.append(100000)

        objective += (gamma ** i) * (pt.T @ Q @ pt + q * (1 - SX.cos(et2)) ** 2)

        # Control inputs
        vt = SX.sym("v{0}".format(i))
        wt = SX.sym("w{0}".format(i))
        ut = vertcat(vt, wt)
        var_list.append(vt)
        var_list.append(wt)
        initial_guess.append(0)
        initial_guess.append(0)
        # ut \in U constraint
        lbx.append(0)
        ubx.append(1)
        lbx.append(-1)
        ubx.append(1)

        objective += (gamma ** i) * (ut.T @ R @ ut)

    # Terminal Cost
    et0 = et[0] + SX.cos(et[2] + ref_states[T][2]) * vt + ref_states[T-1][0] - ref_states[T][0]
    et1 = et[1] + SX.sin(et[2] + ref_states[T][2]) * vt + ref_states[T-1][1] - ref_states[T][1]
    et2 = et[2] + wt + ref_states[i-1][2] - ref_states[i][2]
    objective += et0 ** 2 + et1 ** 2 + q * (1 - SX.cos(et2)) ** 2

    nlp = {}
    nlp['x'] = vertcat(*var_list) # Optimization variable
    nlp['f'] = objective # Objective
    nlp['g'] = vertcat(*constraints)

    # Solve the Problem
    F = nlpsol('F','ipopt',nlp)
    result = F(x0=initial_guess, lbx=lbx, ubx=ubx, lbg=constraint_lower, ubg=constraint_upper)

    u_opt = result['x']
    return np.array([float(u_opt[0]), float(u_opt[1])])

# This function implement the car dynamics
def car_next_state(time_step, cur_state, control, noise = True):
    theta = cur_state[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    f = rot_3d_z @ control
    mu, sigma = 0, 0.04 # mean and standard deviation for (x,y)
    w_xy = np.random.normal(mu, sigma, 2)
    mu, sigma = 0, 0.004  # mean and standard deviation for theta
    w_theta = np.random.normal(mu, sigma, 1)
    w = np.concatenate((w_xy, w_theta))
    if noise:
        return cur_state + time_step*f.flatten() + w
    else:
        return cur_state + time_step*f.flatten()

if __name__ == '__main__':
    # Obstacles in the environment
    obstacles = np.array([[-2,-2,0.5], [1,2,0.5]])
    # Params
    traj = lissajous
    ref_traj = []
    error = 0.0
    car_states = []
    times = []
    # Start main loop
    main_loop = time()  # return time in sec
    # Initialize state
    cur_state = np.array([x_init, y_init, theta_init])
    cur_iter = 0
    T = 2
    # Main loop
    while (cur_iter * time_step < sim_time):
        t1 = time()
        # Get reference state
        cur_time = cur_iter*time_step
        cur_ref = traj(cur_iter)
        next_refs = []
        for i in range(T + 1):
            next_refs.append(traj(cur_iter + i))
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        # TODO: Replace this simple controller with your own controller
        # control = simple_controller(cur_state, cur_ref)
        control = cec_controller(cur_state, next_refs, T)
        print("[v,w]", control)
        ################################################################

        # Apply control input
        next_state = car_next_state(time_step, cur_state, control, noise=True)
        # Update current state
        cur_state = next_state
        # Loop time
        t2 = time()
        print(cur_iter)
        print(t2-t1)
        times.append(t2-t1)
        error = error + np.linalg.norm(cur_state[:2] - cur_ref[:2])
        angle_diff = np.abs((cur_state[2] - cur_ref[2]) % (2 * np.pi))
        error += min(angle_diff, 2 * np.pi - angle_diff)
        cur_iter = cur_iter + 1

    main_loop_time = time()
    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('Average iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('Final error: ', error)

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    visualize(car_states, ref_traj, obstacles, times, time_step, save=True)

