import numpy as np

# Load data from file
data = np.loadtxt("../../ex1data1.txt", delimiter=",")

# Separate into x and y
x = data[:, 0]
y = data[:, 1]

import matplotlib.pyplot as plt

plt.scatter(x, y,s=8, alpha=0.8, marker='x')
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.title("Example 1 Data Scatter Plot")
plt.grid(False)
plt.show()