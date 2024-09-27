import torch
import torch.nn as nn

# Create a simple 2-layer neural network
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = torch.randn(input_size, hidden_size)
        self.w2 = torch.randn(hidden_size, output_size)

    def forward(self, x):
        x_w1 = torch.matmul(x, self.w1)
        h = torch.relu(x_w1)
        return torch.matmul(h, self.w2)

# Generate random data
X = torch.rand(5, 10)
y = torch.rand(5, 1)

# Initialize the model
model = SimpleNN(10, 5, 1)

# Training loop
learning_rate = 0.01
epochs = 100

for epoch in range(epochs + 1):
    # Forward pass
    y_pred = model.forward(X)
    # Compute loss (Mean Squared Error)
    loss = ((y_pred - y) ** 2).mean()

    # Backward pass (manual gradient computation)
    grad_y_pred = 2.0 * (y_pred - y) / y.numel()
    grad_w2 = torch.matmul(torch.relu(torch.matmul(X, model.w1)).t(), grad_y_pred)
    grad_h = torch.matmul(grad_y_pred, model.w2.t())
    grad_w1 = torch.matmul(X.t(), grad_h * (torch.matmul(X, model.w1) > 0).float())

    # Update weights
    model.w1 -= learning_rate * grad_w1
    model.w2 -= learning_rate * grad_w2

    # Print progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print('Training complete!')
