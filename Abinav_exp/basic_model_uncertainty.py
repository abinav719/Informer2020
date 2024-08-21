#The Goal is to experiment whether the model learns uncertainty in regression tasks.
#Take a simple function with 100 points and make the model learn the uncertainty in data
#Add noise to the data and see whether model can predict it correctly

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Generate synthetic data
np.random.seed(42)
x = np.linspace(-1, 1, 100)
y_actual = np.sin(x) * x - np.cos(2 * x)

#Noise addition
def noise_1(max_std):
    linear = np.linspace(0,max_std,100)
    noise_1 = np.random.randn(100)
    return noise_1*linear

y_data = y_actual + noise_1(1)
y_test = y_actual + noise_1(3)

# Convert to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


plt.figure(figsize=(10, 6))
plt.plot(x, y_data, 'b.',label='Data')
plt.plot(x,y_test,'r.',label="Test_data")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.close()
# Define the model
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc_mean = nn.Linear(10, 1)
        self.fc_log_var = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        return mean, torch.exp(log_var)  # Ensure variance is positive

def negative_log_likelihood(y, mean, var):
    return 0.5 * torch.log(var) + 0.5 * (y - mean)**2 / var

# Initialize the model, loss function, and optimizer
model = RegressionModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    mean, var = model(x_tensor)
    y_data = (y_actual + noise_1(1))
    y_tensor = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)
    loss = negative_log_likelihood(y_tensor, mean, var).mean()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Make predictions for training data
model.eval()
with torch.no_grad():
    mean, var = model(x_tensor)

# Convert to numpy for plotting
mean = mean.numpy().flatten()
std = np.sqrt(var.numpy().flatten())

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, y_data, 'b.', label='Data')
plt.plot(x, y_actual, label='Training Data - function')
plt.plot(x, mean, 'r-', label='Predicted Mean')
plt.fill_between(x, mean - 2 * std, mean + 2 * std, color='r', alpha=0.3, label='Uncertainty (2 std)')
plt.plot(x,std,label='std')
plt.title('Regression with Uncertainty')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.close()

# Make predictions for training data
model.eval()
with torch.no_grad():
    mean, var = model(x_tensor)

# Convert to numpy for plotting
mean = mean.numpy().flatten()
std = np.sqrt(var.numpy().flatten())

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, y_test, 'b.', label='Data')
plt.plot(x, y_actual, label='Training Data - function')
plt.plot(x, mean, 'r-', label='Predicted Mean')
plt.fill_between(x, mean - 2 * std, mean + 2 * std, color='r', alpha=0.3, label='Uncertainty (2 std)')
plt.plot(x,std,label='std')
plt.title('Regression with Uncertainty')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
