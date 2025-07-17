dt = 0.05 # Sampling time
N = 30 # Prediction hoirzon
# Control input Constraints
v_min = -1
v_max = 1
w_min = -4
w_max = 4
# Total time of simulation and time steps
T_sim = 10
total_steps_sim = int(T_sim/dt)
# Regularized co-state loss
beta = 0.1