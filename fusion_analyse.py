import pandas
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

"""In this code i took the variance and values of KMB1,kmb2 and informer along with true slip
for the test dataset. This is filtered out of 0 velocity
Then we train a MLP and then deploy it for inference.
For simplicity the initial experiment was trained on test dataset and then the fused value was taken
as csv and plugged inside result analysis directly."""



dataset = pd.read_csv('data_fusion_2_test.csv')
mae_informer = (dataset['informer_val'] - dataset['true_val']).abs().mean()
print('mae_informer',mae_informer)
X = torch.tensor(dataset.drop(columns=['true_val']).values, dtype=torch.float32)
y = torch.tensor(dataset['true_val'].values, dtype=torch.float32).view(-1, 1)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Fit and transform X and y
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
class SimpleRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Model initialization
input_size = X.shape[1]
hidden_size = 10
output_size = 1
model = SimpleRegressor(input_size, hidden_size, output_size)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Put model in evaluation mode
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(X_tensor)
    predictions_numpy = predictions.numpy()
    y_pred_original = scaler_y.inverse_transform(predictions_numpy)
    y_original = scaler_y.inverse_transform(y_tensor.numpy())
    mae = np.mean(np.abs((y_pred_original  - y_original )))
    print(f'Final Mean Absolute Error (MAE): {mae:.4f}')
    y_pred_df = pd.DataFrame(y_pred_original, columns=['best_fusion_val'])

    # Save the DataFrame to CSV
    y_pred_df.to_csv('y_pred_original.csv', index=False)



