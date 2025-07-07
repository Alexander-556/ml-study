# model.py
# The class definition of LinearNormal Class, Analytical approach.

"""Import Libraries and Custom Files"""

# Necessary Libraries
import torch


class LinearNormal:
    # Class Constructor
    def __init__(self):

        # Define coefficient vector 
        # Will hold [b, w]
        self.theta = None  

    # Function: fit()
    # Fit model using normal equation.
    def fit(self, x: torch.Tensor, y: torch.Tensor):
        
        # Grab the length of the 1D array
        n = x.shape[0]
        
        # Add the bias column of 1
        X = torch.cat([torch.ones_like(x), x], dim=1)

        # Transpose the Input X tensor
        Xt = X.T
        # Multiply X transposed by X matrix
        XtX = Xt @ X

        # Calculate inverse of XtX
        XtX_inv = torch.inverse(XtX)

        # Multiply everything together to get theta
        self.theta = XtX_inv @ Xt @ y

    # Function: predict()
    # Predict y values for input x using learned theta.
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.theta is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        b = self.theta[0]
        w = self.theta[1]

        return x * w + b

    # Function: coefficients()
    # Print out the coefficients in a tuple form. 
    def coefficients(self) -> tuple[float, float]:
        if self.theta is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        # Use item() for higher precision and better type
        b = self.theta[0].item()
        w = self.theta[1].item()
        return w, b
