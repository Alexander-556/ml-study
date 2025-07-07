import torch
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.loadtxt("../../ex1data1.txt", delimiter=",")

# reshape make the list of numbers into an array
# -1 means auto dimension
# 1  means one column
x_np = data[:, 0].reshape(-1, 1)
y_np = data[:, 1].reshape(-1, 1)

# Convert to float32 torch tensors
x = torch.from_numpy(x_np).float()
y = torch.from_numpy(y_np).float()

# Add column of ones: X = [1, x]
X = torch.cat([torch.ones_like(x), x], dim=1)  # Shape: (n, 2)

# theta = (X^T X)^(-1) X^T y
Xt = X.T  # Transpose
XtX = Xt @ X
XtX_inv = torch.inverse(XtX)
theta = XtX_inv @ Xt @ y

b = theta[0].item()
w = theta[1].item()
print(f"Model: y = {w:.4f} * x + {b:.4f}")

# Predict using the closed-form model
y_pred = x * w + b

# Plot original data and fitted line
plt.scatter(x_np, y_np, s=10, label="Data")
plt.plot(x_np, y_pred.detach().numpy(), color='red', label="Fitted Line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Least Squares Fit (Normal Equation)")
plt.legend()
plt.grid(True)
plt.show()