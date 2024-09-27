import torch
import torch.nn as nn
import torch.optim as optim

# Create a simple 2-layer neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.fc2(h)

# Generate random data
X = torch.rand(5, 10)
y = torch.rand(5, 1)

# Initialize the model
model = SimpleNN(10, 5, 1)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 100

for epoch in range(epochs):
    # Forward pass
    y_pred = model(X)

    # Compute loss
    loss = criterion(y_pred, y)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print('Training complete!')
