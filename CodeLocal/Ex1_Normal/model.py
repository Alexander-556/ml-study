import torch

class LinearNormal:
    def __init__(self):
        self.theta = None  # Will hold [b, w]
    
    def fit(self, x: torch.Tensor, y: torch.Tensor):
        """Fit model using normal equation."""
        n = x.shape[0]
        X = torch.cat([torch.ones_like(x), x], dim=1)  # Add bias column

        Xt = X.T
        XtX = Xt @ X

        # Use pseudo-inverse for safety
        XtX_inv = torch.inverse(XtX)
        self.theta = XtX_inv @ Xt @ y
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict y values for input x using learned theta."""
        if self.theta is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        b = self.theta[0]
        w = self.theta[1]

        return x * w + b

    def coefficients(self) -> tuple[float, float]:
        if self.theta is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        b = self.theta[0].item()
        w = self.theta[1].item()
        return w, b
