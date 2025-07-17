import matplotlib.pyplot as plt
import numpy as np

# x-axis: prediction horizon
time_steps = np.array([15, 20, 25, 30, 35])

# y-axis: values from experiments (may vary depends on computer)
mpc_a = 1000*np.array([0.0617, 0.1025, 0.1573, 0.2266, 0.3146])
mpc_b = 1000*np.array([0.0665, 0.1118, 0.1758, 0.2689, 0.3936])
mpc_c = 1000*np.array([0.0677, 0.1152, 0.1815, 0.275, 0.4169])
a = mpc_a + mpc_b + mpc_c
mpc_mean = (mpc_a + mpc_b + mpc_c) / 3
mpc_mean_log = np.log(mpc_mean)

ncr_a = 1000*np.array([0.0016, 0.0016, 0.0016, 0.0018, 0.0016])
ncr_b = 1000*np.array([0.0016, 0.0017, 0.0016, 0.0016, 0.0016])
ncr_c = 1000*np.array([0.0016, 0.0016, 0.0016, 0.0016, 0.0016])
ncr_mean = (ncr_a + ncr_b + ncr_c) / 3
ncr_mean_log = np.log(ncr_mean)

# Plot the results and save the figure
plt.figure(figsize=(12, 6))
# Plot mpc trajectories
plt.subplot(1, 2, 1)
plt.plot(time_steps, mpc_a, marker='o', linestyle='-', dashes=[3, 1], label=r"$MPC$", markersize=15, linewidth=5)
plt.xlabel("Prediction horizon (steps)", fontsize=20, fontweight='bold')
plt.ylabel("Time (ms)", fontsize=20, fontweight='bold')
plt.legend(fontsize=24)
plt.grid(True)
plt.xticks(time_steps, fontsize=24, fontweight='bold')
plt.yticks(fontsize=24, fontweight='bold')

plt.subplot(1, 2, 2)
plt.plot(time_steps, mpc_mean_log, marker='o', linestyle='-', dashes=[3, 1], label=r"$MPC$", markersize=15, linewidth=5)
plt.plot(time_steps, ncr_mean_log, marker='o', linestyle='-', dashes=[3, 1], label=r"$NCR$", markersize=15, linewidth=5)
plt.xlabel("Prediction horizon (steps)", fontsize=20, fontweight='bold')
plt.ylabel("ln(Time)", fontsize=20, fontweight='bold')
plt.legend(fontsize=24)
plt.grid(True)
plt.xticks(time_steps, fontsize=24, fontweight='bold')
plt.yticks(fontsize=24, fontweight='bold')
plt.tight_layout()
output_dir = f"./figs/time_per_step.png"
plt.savefig(output_dir, dpi=300)
# plt.show()