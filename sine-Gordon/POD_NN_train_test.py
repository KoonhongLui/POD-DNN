import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.sandbox.distributions.sppatch import expect
from torch.utils.data import DataLoader, TensorDataset, random_split

class POD_NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(POD_NN, self).__init__()
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

class ZScoreNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: torch.Tensor):
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0)
        self.std[self.std == 0] = 1.0  # 防止除以0

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        try:
            return (data - self.mean) / self.std
        except:
            std = self.std.to(device)
            mean = self.mean.to(device)
            return (data - mean) / std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        try:
            return data * self.std + self.mean
        except:
            std = self.std.to(device)
            mean = self.mean.to(device)
            return data * std + mean

torch.manual_seed(1111)
np.random.seed(1111)
generator = torch.Generator().manual_seed(1111)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

n_s = 100
n_data = 1024
eps_POD = 0.01

data = np.load(f"./data_set_ns{n_s}_epsPOD{eps_POD}_ndata{n_data}.npz")
X = data['X'].astype(np.float32)
Y = data['Y'].astype(np.float32)
U_hf = data['U_hf'].astype(np.float32).reshape(n_data, -1) #n_data, N, N_t  -> n_data, N*N_t
n_data = X.shape[0]
input_dim = X.shape[1]
output_dim = Y.shape[1]

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

train_indices = train_dataset.indices
normalizer = ZScoreNormalizer()
normalizer.fit(Y_tensor[train_indices])
Y_tensor = normalizer.transform(Y_tensor)

dataset = TensorDataset(X_tensor, Y_tensor, U_hf_tensor)
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size], generator=generator
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

hidden_dim = 500
model = POD_NN(input_dim, hidden_dim, output_dim).to(device)

criterion = nn.MSELoss()
lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=lr)
try:
    model.load_state_dict(torch.load(f'./POD_NN_best_model_epsPOD{eps_POD}_lr{lr}_zscore.pth'))
except:
    num_epochs = 2000
    best_val_loss = float('inf')
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x, y, _ in train_loader:
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
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                x = model(x)
                loss = criterion(x, y)
                val_loss += loss.item() * x.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, f'./POD_NN_best_model_epsPOD{eps_POD}_lr{lr}_zscore.pth')

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.8f}, "
                  f"Val Loss: {val_loss:.8f}")
    model.load_state_dict(best_model_state)

V = np.load(f'./POD_mat_ns{n_s}_epsPOD{eps_POD}.npy').astype(np.float32)
V_tensor = torch.from_numpy(V)
V_tensor = V_tensor.to(device) # N*N_t, d
model.eval()
test_loss = 0.0
relative_l2_error = 0.0
with torch.no_grad():
    for x, y, u_hf in test_loader:
        x, y, u_hf = x.to(device), y.to(device), u_hf.to(device)
        x = model(x)   # batch_size, d
        loss = criterion(x, y)
        test_loss += loss.item() * x.size(0)

        x = normalizer.inverse_transform(x)
        x = torch.matmul(V_tensor, x.T).T   # batch_size, N*N_t
        relative_l2_error += Relative_l2_error(y_pred = x, y_true = u_hf).item() * x.size(0)
test_loss = test_loss / len(test_loader.dataset)
relative_l2_error = relative_l2_error / len(test_loader.dataset)
print(f"Test Loss: {test_loss:.8f}")
print(f"Relative l2 Error: {relative_l2_error:.8f}")
# eps_POD 0.01:
# Train Time: 15 min
# Test Loss: 3058.00537109
# Relative l2 Error: 0.13736392

# d 15 lr 0.0001
# Test Loss: 5315.68363953
# Relative l2 Error: 0.13824874

# d 15 lr 0.001
# Test Loss: 260.70863628
# Relative l2 Error: 0.04001865

# d 15 lr 0.01
# Test Loss: 214.07849312
# Relative l2 Error: 0.04250938

# d 15 lr 0.0001 zscore
# Test Loss: 0.05204761
# Relative l2 Error: 0.03601874

######## eps_POD lr 0.0001 zscore
# Test Loss: 0.14759446
# Relative l2 Error: 0.03717648

# eps_POD lr 0.001 zscore
# Test Loss: 0.03215166
# Relative l2 Error: 0.03634274
# 0.0014705467224121094