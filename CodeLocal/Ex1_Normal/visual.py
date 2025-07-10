# visual.py
# Plot the given data and the predicted values. 

"""Import Libraries and Custom Files"""
# Necessary Libraries
import matplotlib.pyplot as plt
import numpy as np

def plot_regression(x_np, y_np, y_pred_np):

    # Scatter Plot Data Points
    plt.scatter(
        x_np,
        y_np,
        s=8,
        label="Data",
    )

    # Plot Fitted Line
    plt.plot(
        x_np,
        y_pred_np,
        color="red",
        label="Fitted Line",
    )
    
    #Plot labels
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Least Squares Fit (Normal Equation)")
    plt.legend()
    plt.grid(True)
    plt.savefig("./CodeLocal/Ex1_Normal/Plots/LinearReg.png")
    plt.close()
