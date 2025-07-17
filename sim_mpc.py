import numpy as np
from mpc import MPC
from config import dt, T_sim, total_steps_sim, N, v_max, v_min, w_max, w_min
from utils import plot_traj
import time

def simulation(x_0, traj_ref, h):
    state_traj = [x_0]
    control_traj = []

    x_k = x_0
    # Simulation loop
    for k in range(total_steps_sim):
        # Calculate control input
        ref_traj_segment = traj_ref[:, k:k + N + 1]
        u_k = controller.get_control_input(x_k, ref_traj_segment, h)
        print(f'MPC N={N} - timestep: {k} finished')
        
        control_traj.append(u_k)
        x_k = controller.rk4(x_k, u_k)
        x_k = x_k.full().flatten().tolist()
        state_traj.append(x_k)
    # Convert state and control trajectories to numpy arrays for plotting
    state_traj = np.array(state_traj)
    control_traj = np.array(control_traj)
    return state_traj, control_traj

if __name__ == '__main__':
    state_0a = np.array([[-1.16, 1.37, -1.79]])
    state_0b = np.array([[-5.24, 4.11, 2.72]])
    state_0c = state_0b

    # Change case to a, b or c here
    initial_state_option = 'c'
    if initial_state_option == 'a':
        state_0 = state_0a
    elif initial_state_option == 'b':
        state_0 = state_0b
    else:
        state_0 = state_0c
        # Define reference state
        x_ref = 1; y_ref = 1; theta_ref = 0
    
    
    t_span = np.arange(0, T_sim + N * dt, dt)
    traj_ref = np.zeros((3, total_steps_sim + N))
    if initial_state_option == 'c':
        traj_ref[0,:] = x_ref
        traj_ref[1,:] = y_ref
        traj_ref[2,:] = theta_ref

    # Initialize controller and initial state
    controller = MPC(dt, N, v_min, v_max, w_min, w_max)
    robot_x0 = state_0[0,0]; robot_y0 = state_0[0,1]; robot_theta0 = state_0[0,2]
    x_0 = np.array([robot_x0, robot_y0, robot_theta0])
    h = 50 # Terminal cost coefficient
    
    # Start timing
    start_time = time.time()
    state_traj, control_traj = simulation(x_0, traj_ref, h)
    # End timing
    end_time = time.time()
    
    # Compute total time taken
    execution_time = end_time - start_time
    time_per_step = execution_time / total_steps_sim
    print(f"Simulation executed in {execution_time:.2f}s, time per step: {time_per_step:.4f}s")

    # Plot state and control trajectories
    plot_traj(state_mpc=state_traj, u_mpc=control_traj, 
              time=t_span, h=h, option=initial_state_option)
    
    x_mpc = state_traj[:, 0]
    y_mpc = state_traj[:, 1]
    theta_mpc = state_traj[:,2]
    v_mpc = control_traj[:, 0, 0]
    w_mpc = control_traj[:, 1, 0]

    # Compute trajectory gradients (numerical derivatives)
    x_traj_grad = np.gradient(x_mpc, dt)
    y_traj_grad = np.gradient(y_mpc, dt)
    theta_traj_grad = np.gradient(theta_mpc, dt)

    v_traj_grad = np.gradient(v_mpc, dt)
    w_traj_grad = np.gradient(w_mpc, dt)

    # Compute mean squared derivative
    x_traj_msd = np.mean(x_traj_grad ** 2)
    y_traj_msd = np.mean(y_traj_grad ** 2)
    theta_traj_msd = np.mean(theta_traj_grad ** 2)
    avg_state_msd = (x_traj_msd + y_traj_msd + theta_traj_msd) / 3
    print(f"Average State Trajectory Mean Squared derivatives {avg_state_msd:.2f}")

    v_traj_msd = np.mean(v_traj_grad ** 2)
    w_traj_msd = np.mean(w_traj_grad ** 2)
    avg_u_msd = (v_traj_msd + w_traj_msd) / 2
    print(f"Average Control Input Trajectory Mean Squared derivatives {avg_u_msd:.2f}")

    # Calculate absolute convergence error
    if initial_state_option == 'c':
        abs_convergence_err = abs(x_mpc[-1] - x_ref) + abs(y_mpc[-1] - y_ref) + abs(theta_mpc[-1] - theta_ref)
    else:
        abs_convergence_err = abs(x_mpc[-1]) + abs(y_mpc[-1]) + abs(theta_mpc[-1])
    print(f'Final state: [{x_mpc[-1]:.2f}; {y_mpc[-1]:.2f}; {theta_mpc[-1]:.2f}]')
    print(f'Absolute convergence error: {abs_convergence_err:.2f}')

