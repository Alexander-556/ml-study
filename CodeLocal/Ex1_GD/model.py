# model_gd.py
import torch
from typing import Optional

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w: Optional[torch.Tensor] = None
        self.b: Optional[torch.Tensor] = None
        self.loss_history = []

    def fit(self, x: torch.Tensor, y: torch.Tensor):
        m = x.shape[0]

        # Initialize parameters
        self.w = torch.randn(1, requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)

        for epoch in range(self.epochs):
            y_pred = x * self.w + self.b
            loss = (1 / m) * torch.sum((y_pred - y) ** 2)

            loss.backward()

            with torch.no_grad():
                if self.w.grad is not None:
                    
                    self.w -= self.lr * self.w.grad

                    if self.w.grad is not None:
                        self.w.grad.zero_()

                if self.b.grad is not None:

                    self.b -= self.lr * self.b.grad

                    if self.w.grad is not None:
                        self.w.grad.zero_()

            self.loss_history.append(loss.item())

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, w: {self.w.item():.4f}, b: {self.b.item():.4f}")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.w is None or self.b is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return x * self.w + self.b

    def coefficients(self) -> tuple[float, float]:
        if self.w is None or self.b is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self.w.item(), self.b.item()
    
