import numpy as np

# Define parameters
dt = 0.01
T_simulation = 10
steps = int(T_simulation / dt)

# Initialize arrays
X_space = np.zeros((2, steps))
X0 = np.array([[5], [0]])
Ft = np.array([[1, dt], [0, 1]])
Bt = np.array([[(dt**2) / 2.0], [dt]])
ut = np.array([10 / 1.0])

# Run simulation
Xact = X0
for k in range(steps):
    # Update Xact
    Xact = np.dot(Ft, Xact) + np.dot(Bt, ut) + np.random.randn(2, 1)  # Add process noise
    X_space[:, k] = Xact.flatten()

print(X_space)