import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
plt.ioff()
from torch.utils.data import DataLoader, TensorDataset, Subset

class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x


def Relative_l2_error(y_pred, y_true):
    l2_error = torch.norm(y_pred - y_true, p=2, dim=1)
    l2_norm = torch.norm(y_true, p=2, dim=1)
    eps = 1e-8
    relative_l2 = l2_error / (l2_norm + eps)
    mean_relative_l2 = torch.mean(relative_l2)
    return mean_relative_l2

torch.manual_seed(1111)
np.random.seed(1111)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

discretization = np.load('./discretization.npz')
XYZ = discretization['xyz'].astype(np.float32)
BDY = discretization['bdy'].astype(np.float32)
NODE = np.concatenate((XYZ, BDY))

eps_POD = 0.03
n_data = 500
data = np.load(f'./data_set_ndata{n_data}_epsPOD{eps_POD}.npz')
X = data['X'].astype(np.float32)
Y = data['Y'].astype(np.float32)
U_hf = data['U_hf'].astype(np.float32)
n_data = X.shape[0]
input_dim = X.shape[1]
output_dim = U_hf.shape[1]

X_tensor = torch.from_numpy(X)
Y_tensor = torch.from_numpy(Y)
U_hf_tensor = torch.from_numpy(U_hf)

dataset = TensorDataset(X_tensor, Y_tensor, U_hf_tensor)
train_size = 300
val_size = 100
test_size = 100
batch_size = 10
train_dataset = Subset(dataset, range(train_size))
val_dataset = Subset(dataset, range(train_size, train_size+val_size))
test_dataset = Subset(dataset, range(train_size+val_size, n_data))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

hidden_dim = 500
model = NN(input_dim, hidden_dim, output_dim).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
try:
    model.load_state_dict(torch.load('./NN_best_model.pth'))
except:
    num_epochs = 2000
    best_val_loss = float('inf')
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x, _, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = model(x)
            loss = criterion(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _, y in val_loader:
                x, y = x.to(device), y.to(device)
                x = model(x)
                loss = criterion(x, y)
                val_loss += loss.item() * x.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, './NN_best_model.pth')

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.8f}, "
                  f"Val Loss: {val_loss:.8f}")
    model.load_state_dict(best_model_state)

model.eval()
test_loss = 0.0
relative_l2_error = 0.0
with torch.no_grad():
    for x, _, y in test_loader:
        x, y = x.to(device), y.to(device)
        x = model(x)
        loss = criterion(x, y)
        test_loss += loss.item() * x.size(0)
        relative_l2_error += Relative_l2_error(y_pred = x, y_true = y).item() * x.size(0)
test_loss = test_loss / len(test_loader.dataset)
relative_l2_error = relative_l2_error / len(test_loader.dataset)
print(f"Test Loss: {test_loss:.8f}")
print(f"Relative l2 Error: {relative_l2_error:.8f}")


sample_idx = 1
u_pred = x[sample_idx, :].detach().cpu().numpy()
u_true = y[sample_idx, :].detach().cpu().numpy()
u_error = np.abs(u_pred-u_true)
x = NODE[:, 0]
y = NODE[:, 1]
z = NODE[:, 2]

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(131, projection='3d')
scatter1 = ax1.scatter(x, y, z, c=u_pred, cmap='viridis', s=10)
cbar1 = fig.colorbar(scatter1, ax=ax1, shrink=0.6, pad=0.1)
cbar1.set_label("Values")

ax2 = fig.add_subplot(132, projection='3d')
scatter2 = ax2.scatter(x, y, z, c=u_true, cmap='viridis', s=10)
cbar2 = fig.colorbar(scatter2, ax=ax2, shrink=0.6, pad=0.1)
cbar2.set_label("Values")

ax3 = fig.add_subplot(133, projection='3d')
scatter3 = ax3.scatter(x, y, z, c=u_error, cmap='viridis', s=10)
cbar3 = fig.colorbar(scatter3, ax=ax3, shrink=0.6, pad=0.1)
cbar3.set_label("Values")

plt.tight_layout()
plt.savefig('NN_plot_error.png')
plt.show()
# Test Loss: 0.00035578
# Relative l2 Error: 0.09717311
# 0.0005174684524536133


