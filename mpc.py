import casadi as ca
import numpy as np

class MPC:
    def __init__(self, dt, N, v_min, v_max, w_min, w_max):
        # System parameters
        self.dt = dt
        self.N = N
        self.v_min = v_min
        self.v_max = v_max
        self.w_min = w_min
        self.w_max = w_max

    def dynamics(self, x, u):
        # Unicycle model dynamics
        theta = x[2]
        V, omega = u[0], u[1]
        x_dot = V * ca.cos(theta)
        y_dot = V * ca.sin(theta)
        theta_dot = omega
        return ca.vertcat(x_dot, y_dot, theta_dot)

    def rk4(self, x, u):
        # RK4 integration step for dynamics
        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + self.dt / 2 * k1, u)
        k3 = self.dynamics(x + self.dt / 2 * k2, u)
        k4 = self.dynamics(x + self.dt * k3, u)
        x_next = x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next

    def get_control_input(self, current_state, ref_traj, h):
        # Define control decision variables (U only, for shooting method)
        U = ca.MX.sym("U", 2, self.N)

        # Objective and cost weights
        obj = 0
        Q = np.diag([10, 10, 10])  # State cost weights
        R = np.diag([1, 1])   # Control cost weights
        H = h*Q

        # Constraints for control inputs only
        lbu = [self.v_min, self.w_min] * self.N
        ubu = [self.v_max, self.w_max] * self.N

        # Initial state
        x_k = current_state
        x_k = ca.reshape(x_k, -1, 1)
        # Build the objective by simulating forward using rk4 and accumulating cost
        for k in range(self.N):
            # Get control input at step k
            u_k = U[:, k]
            ref_k = ref_traj[:, k]
            ref_k = ca.reshape(ref_k, -1, 1)

            # Compute the cost
            obj += ca.mtimes((x_k - ref_k).T, Q @ (x_k - ref_k)) + ca.mtimes(u_k.T, R @ u_k)

            # Forward propagate the state using rk4 with the control input
            x_k = self.rk4(x_k, u_k)
            
        # Add terminal cost
        obj += ca.mtimes((x_k - ref_traj[:, k+1]).T, H @ (x_k - ref_traj[:, k+1]))

        # Define the optimization problem
        nlp = {'f': obj, 'x': ca.reshape(U, -1, 1)}
        opts = {'ipopt.print_level':1,'print_time': 0}
        solver = ca.nlpsol('S', 'ipopt', nlp, opts)

        # Solve the optimization problem
        sol = solver(lbx=lbu, ubx=ubu)
        u_opt_traj = sol["x"].full()
        u = u_opt_traj[:2]
        return u  # Return only the first control action
    
