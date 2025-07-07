import numpy as np
import torch
from model import LinearRegressionGD
from visual import plot_regression

def main():
    # Load and prepare data
    data = np.loadtxt("../../Docs/ex1data1.txt", delimiter=",")
    x_np = data[:, 0].reshape(-1, 1)
    y_np = data[:, 1].reshape(-1, 1)

    x = torch.from_numpy(x_np).float()
    y = torch.from_numpy(y_np).float()

    # Train using gradient descent
    model = LinearRegressionGD(learning_rate=0.01, epochs=1000)
    model.fit(x, y)

    # Predict
    y_pred = model.predict(x)
    w, b = model.coefficients()
    print(f"Learned model: y = {w:.4f} * x + {b:.4f}")

    # Plot result
    plot_regression(x_np, y_np, y_pred.detach().numpy())

if __name__ == "__main__":
    main()
