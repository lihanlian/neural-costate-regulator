import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from config import T_sim, total_steps_sim, dt
from utils import solve_qp, rk4, save_animation
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Neural Network Model
class CoNN(nn.Module):
    def __init__(self, prediction_horizon):
        super(CoNN, self).__init__()
        self.prediction_horizon = prediction_horizon
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3*prediction_horizon)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, self.prediction_horizon, 3)
        return x
    
# Specify prediction horizon n and model names
n = 30
h = 50
epoch = 50
seed = 0
model = CoNN(n).to(device)

model_name = f'ncr_N{n}_h{h}_seed_{seed}_e{epoch}.pth' # full horizon abs lambda loss
print(f'load model: {model_name}')
model.load_state_dict(torch.load(f'./model/{model_name}'))
model.eval()

# Define the initial condition and total time steps
state_0a = np.array([[-1.16, 1.37, -1.79]])
state_0b = np.array([[-5.24, 4.11, 2.72]])
state_0c = state_0b
t_span = np.linspace(0, T_sim, total_steps_sim+1)

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


# Simulate the state trajectory using the CoNN-based controller in a feedback loop (without disturbance)
state_traj_undisturbed = [state_0]
u_traj_undisturbed = []
state_k = state_0

# Start timing
start_time = time.time()
costate_trajectory = []
for i in range(total_steps_sim):
    # Predict the co-state trajectory using the trained CoNN
    state_k_tensor = torch.tensor(state_k, dtype=torch.float32, device=device)
    if initial_state_option == 'c':
        error_k = np.array([[state_k[0,0] - x_ref, state_k[0,1] - y_ref, state_k[0,2] - theta_ref]])
        error_k_tensor = torch.tensor(error_k, dtype=torch.float32, device=device)
        costate_traj_k_hat = model(error_k_tensor).cpu().detach().numpy()
    else:
        costate_traj_k_hat = model(state_k_tensor).cpu().detach().numpy()

    costate_trajectory.append(costate_traj_k_hat)
    lambda1_k_hat = costate_traj_k_hat[0,0,0]
    lambda2_k_hat = costate_traj_k_hat[0,0,1]
    lambda3_k_hat = costate_traj_k_hat[0,0,2]
    
    # Impose input constraints
    v, omega = solve_qp(lambda1=lambda1_k_hat, lambda2=lambda2_k_hat, 
                 lambda3=lambda3_k_hat, theta=state_k_tensor[0,2].cpu().detach().numpy())
    v = float(v)
    omega = float(omega)

    u_k = np.array([v, omega])
    u_traj_undisturbed.append(u_k)

    state_k = rk4(state_k, u_k)
    state_traj_undisturbed.append(state_k)

# End timing
end_time = time.time()
# Compute time NCR takes
execution_time = end_time - start_time
time_per_step = execution_time / total_steps_sim
print(f"Simulation executed in {execution_time:.2f}s, time per step: {time_per_step:.4f}s")

state_traj_undisturbed = np.array(state_traj_undisturbed).squeeze(1)
u_traj_undisturbed = np.array(u_traj_undisturbed)
x_traj = state_traj_undisturbed[:,0]
y_traj = state_traj_undisturbed[:,1]
theta_traj = state_traj_undisturbed[:,2]
v_traj = u_traj_undisturbed[:,0]
w_traj = u_traj_undisturbed[:,1]
final_state_undisturbed = state_traj_undisturbed[-1]

# Compute trajectory gradients (numerical derivatives)
x_traj_grad = np.gradient(x_traj, dt)
y_traj_grad = np.gradient(y_traj, dt)
theta_traj_grad = np.gradient(theta_traj, dt)

v_traj_grad = np.gradient(v_traj, dt)
w_traj_grad = np.gradient(w_traj, dt)

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

# Plot the results and save the figure
plt.figure(figsize=(12, 6))

# Plot x trajectories
plt.subplot(1, 2, 1)
plt.plot(t_span, x_traj, linestyle='-', dashes=[3, 1], label=r"$x_{ncr}$", linewidth=5)
plt.plot(t_span, y_traj, linestyle='-', dashes=[3, 1], label=r"$y_{ncr}$", linewidth=5)
plt.plot(t_span, theta_traj, linestyle='-', dashes=[3, 1], label=r"$\theta_{ncr}$", linewidth=5)
plt.xlabel("Time (s)", fontsize=20, fontweight='bold')
plt.ylabel("State Trajectory", fontsize=20, fontweight='bold')
plt.legend(fontsize=28)
plt.grid(True)
plt.xticks(fontsize=24, fontweight='bold')
plt.yticks(fontsize=24, fontweight='bold')

# Plot u trajectories
plt.subplot(1, 2, 2)
plt.plot(t_span[:-1], v_traj, linestyle='-', dashes=[3, 1], label=r"$v_{ncr}$", linewidth=5)
plt.plot(t_span[:-1], w_traj, linestyle='-', dashes=[3, 1], label=r"$w_{ncr}$", linewidth=5)
plt.xlabel("Time (s)", fontsize=20, fontweight='bold')
plt.ylabel("Control Input", fontsize=20, fontweight='bold')
plt.legend(fontsize=28)
plt.grid(True)
plt.xticks(fontsize=24, fontweight='bold')
plt.yticks(fontsize=24, fontweight='bold')
plt.tight_layout()
output_dir = f"./figs/ncr_N{n}_h{h}_{initial_state_option}.png"
plt.savefig(output_dir, dpi=300)
plt.close()

print(f"Figure saved to {output_dir}")
if initial_state_option == 'c':
    abs_convergence_err = abs(x_traj[-1] - x_ref) + abs(y_traj[-1] - y_ref) + abs(theta_traj[-1] - theta_ref)
else:
    abs_convergence_err = abs(x_traj[-1]) + abs(y_traj[-1]) + abs(theta_traj[-1])
print(f'Final state: [{x_traj[-1]:.2f}; {y_traj[-1]:.2f}; {theta_traj[-1]:.2f}]')
print(f'Absolute convergence error: {abs_convergence_err:.2f}')

save_animation(t_span, x_traj, y_traj, theta_traj, 
               costate_trajectory, initial_state_option)


