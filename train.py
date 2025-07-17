import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchdiffeq import odeint # Use odeint for integration
import numpy as np
from utils import UnicycleDynamics, set_seed
from config import dt, beta
from torchdiffeq import odeint


# Step 2: Create Dataset Class
class InitialStateDataset(Dataset):
    def __init__(self, initial_states):
        # Store the initial states as a PyTorch tensor
        self.initial_states = torch.tensor(initial_states, dtype=torch.float32)

    def __len__(self):
        return len(self.initial_states)

    def __getitem__(self, idx):
        # Return a single state
        return self.initial_states[idx]

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


# Training Setup
def train_network(initial_states, n, h, q1, q2, q3, r1, r2, model_save_path, batch_size=1, epochs=50, lr=2e-4):

    dataset = InitialStateDataset(initial_states)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    # Time span for integration
    t_span = torch.tensor([0, dt], dtype=torch.float32, device=device)
    ode_solver = UnicycleDynamics()

    # Initialize NN
    model = CoNN(n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    # Define cost matrices
    Q = torch.diag(torch.tensor([q1, q2, q3], device=device))  # State cost
    R = torch.diag(torch.tensor([r1, r2], device=device))      # Control input cost
    H =  h*Q                                                   # Terminal cost

    for epoch in range(epochs):     
        epoch_loss = 0
        epoch_lambda_loss = 0
        # Iterate over all initial conditions
        for state_0 in dataloader:

            optimizer.zero_grad()
            state_0 = state_0.to(device)

            costate_traj_k_hat = model(state_0)   # Predicted co-state trajectory starting at time k
            state_k = state_0
            L_stage = 0
            L_terminal = 0
            lambda_cost = 0

            for i in range(n):
                lambda1_i = costate_traj_k_hat[0,i,0]
                lambda2_i = costate_traj_k_hat[0,i,1]
                lambda3_i = costate_traj_k_hat[0,i,2]
                theta_i = state_k[0, 2]

                v_opt = -0.5 * (lambda1_i * torch.cos(theta_i) + lambda2_i * torch.sin(theta_i))
                w_opt = -0.5 * lambda3_i
                u_opt = torch.cat([v_opt.unsqueeze(0), w_opt.unsqueeze(0)], dim=0).unsqueeze(0)

                # Compute stage cost using matrices
                state_cost = state_k @ Q @ state_k.T   # Quadratic cost for state
                control_cost = u_opt @ R @ u_opt.T     # Quadratic cost for control inputs
                
                lambda_cost += torch.abs(lambda1_i) + torch.abs(lambda2_i) + torch.abs(lambda3_i)
                L_stage += state_cost + control_cost

                # Solve the initial value problem using odeint (Step simulation forward by dt)
                x_and_u = torch.cat([state_k, u_opt], dim=1).to(device)
                result = odeint(ode_solver, x_and_u, t_span, method='rk4')
                state_k = result[-1,:,:3]  

            # Compute L_terminal
            L_terminal = state_k @ H @ state_k.T
            # Backpropagation
            loss = L_stage + L_terminal + beta*lambda_cost
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_lambda_loss += lambda_cost.item()
            

        avg_loss = epoch_loss / len(dataset)
        avg_lambda_loss = epoch_lambda_loss / len(dataset)
        print(f"********Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.2f}, Lambda Loss {avg_lambda_loss:.2f}********")

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    seed = 0
    set_seed(seed)

    n = 30 # Prediction horizon
    h = 50 # Terminal cost coefficient
    epoch = 50
    os.makedirs("./model", exist_ok=True)

    # Step 1: Generate 1000 combinations of (x, y, theta)
    x_range = np.linspace(-2, 2, 10)
    y_range = np.linspace(-2, 2, 10)
    theta_range = np.linspace(-2, 2, 10)

    # Create a grid of all combinations
    x, y, theta = np.meshgrid(x_range, y_range, theta_range)
    initial_states = np.vstack([x.ravel(), y.ravel(), theta.ravel()]).T

    # Randomly shuffle the data set
    np.random.shuffle(initial_states)
    
    # Train the model
    q1 = 10.0; q2 = 10.0; q3 = 10.0; r1 = 1.0; r2 = 1.0
    model_save_path = f"./model/ncr_N{n}_h{h}_seed_{seed}_e{epoch}.pth"
    train_network(initial_states, n, h, q1, q2, q3, r1, r2, model_save_path, epochs=epoch, lr=1e-3)
