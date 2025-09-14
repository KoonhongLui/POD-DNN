import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

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
generator = torch.Generator().manual_seed(1111)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

n_s = 100
n_data = 1024
d = 15

data = np.load(f"./data_set_ns{n_s}_d{d}_ndata{n_data}.npz")
X = data['X'].astype(np.float32)
Y = data['Y'].astype(np.float32)
U_hf = data['U_hf'].astype(np.float32).reshape(n_data, -1)  #n_data, N, N_t  -> n_data, N*N_t
n_data = X.shape[0]
input_dim = X.shape[1]
output_dim = U_hf.shape[1]

X_tensor = torch.from_numpy(X)
Y_tensor = torch.from_numpy(Y)
U_hf_tensor = torch.from_numpy(U_hf)

dataset = TensorDataset(X_tensor, Y_tensor, U_hf_tensor)
train_size = 768
val_size = 128
test_size = 128
batch_size = 16
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size], generator=generator
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

hidden_dim = 500
model = NN(input_dim, hidden_dim, output_dim).to(device)

criterion = nn.MSELoss()
lr=0.0001
optimizer = optim.Adam(model.parameters(), lr=lr)
try:
    model.load_state_dict(torch.load(f'./NN_best_model_lr{lr}.pth'))
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
            torch.save(best_model_state, f'./NN_best_model_lr{lr}.pth')

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
# Train Time: 124 min

# lr 0.0001
# Test Loss: 0.02726995
# Relative l2 Error: 0.03589715

# lr 0.001
# Test Loss: 0.00530680
# Relative l2 Error: 0.01889965

# lr 0.0001 zscore
# Test Loss: 0.00168936
# Relative l2 Error: 0.03238899

# lr 0.001 zscore
# Test Loss: 0.00031026
# Relative l2 Error: 0.01445065
# 0.0007102584838867187

