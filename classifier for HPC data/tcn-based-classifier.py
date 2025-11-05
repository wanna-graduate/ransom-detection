import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import time
from pytorch_tcn import TCN
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
best_model_path = "best_tcn_model65"
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
benign_data = pd.read_csv('benign-80-nor.csv')
malicious_data = pd.read_csv('mal-nor.csv')
benign_val_data = pd.read_csv('benign-20-nor.csv')
malicious_val_data = pd.read_csv('unknown-nor.csv')

# Data preprocessing functions
def batch_split_k_fold(data, batch_size, k, k_fold=10):
    """Perform k-fold splitting on data."""
    total_batches = len(data) // batch_size
    train_data, test_data = [], []

    for i in range(total_batches):
        batch = data[i * batch_size:(i + 1) * batch_size]
        fold_size = batch_size // k_fold
        test_start_idx = k * fold_size
        test_end_idx = (k + 1) * fold_size

        # Define training and test splits
        test_batch = batch[test_start_idx:test_end_idx]
        train_batch = np.concatenate((batch[:test_start_idx], batch[test_end_idx:]))
        train_data.append(train_batch)
        test_data.append(test_batch)

    return np.concatenate(train_data), np.concatenate(test_data)

def create_sliding_windows(data, window_size, step=20):
    """Create sliding windows from data."""
    return np.array([data[i:i + window_size] for i in range(0, len(data) - window_size, step)])

def create_dataloader(X, y, batch_size):
    """Create DataLoader from input and target arrays."""
    X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def count_parameters(model):
    """Returns the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Global parameters
window_size = 60
batch_size = 5400
input_size = 5
num_channels = [32,32,32,32]
kernel_size = 3
dropout = 0.1
batch_size = 256
num_epochs = 150
learning_rate = 0.005
k = 2  # Fold index for k-fold cross-validation

# Train-test split for main data
benign_train, benign_test = batch_split_k_fold(benign_data, 5400, k)
malicious_train, malicious_test = batch_split_k_fold(malicious_data, 5400, k)

# Create sliding windows
benign_train_windows = create_sliding_windows(benign_train, window_size)
benign_test_windows = create_sliding_windows(benign_test, window_size)
malicious_train_windows = create_sliding_windows(malicious_train, window_size)
malicious_test_windows = create_sliding_windows(malicious_test, window_size)
benign_val_windows = create_sliding_windows(benign_val_data.to_numpy(), window_size)
malicious_val_windows = create_sliding_windows(malicious_val_data.to_numpy(), window_size)

# Combine and prepare inputs
X_train = np.concatenate((benign_train_windows, malicious_train_windows), axis=0)
y_train = np.concatenate((np.zeros(len(benign_train_windows)), np.ones(len(malicious_train_windows))), axis=0)
X_test = np.concatenate((benign_test_windows, malicious_test_windows), axis=0)
y_test = np.concatenate((np.zeros(len(benign_test_windows)), np.ones(len(malicious_test_windows))), axis=0)
X_val = np.concatenate((benign_val_windows, malicious_val_windows), axis=0)
y_val = np.concatenate((np.zeros(len(benign_val_windows)), np.ones(len(malicious_val_windows))), axis=0)

# Shuffle data
X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_test, y_test = shuffle(X_test, y_test, random_state=42)
X_val, y_val = shuffle(X_val, y_val, random_state=42)

# Data loaders
train_loader = create_dataloader(X_train, y_train, batch_size)
test_loader = create_dataloader(X_test, y_test, batch_size)
val_loader = create_dataloader(X_val, y_val, batch_size)

# TCN Model
class TCNModel(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        self.tcn = TCN(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            use_skip_connections=False,
            dilations=[1,2,4,8],          
            dropout=dropout,
            causal=False
        )
        self.fc1 = nn.Linear(num_channels[-1], 50)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        tcn_output = self.tcn(x)
        max_pool_out, _ = torch.max(tcn_output, dim=2)  
        hidden = torch.relu(self.fc1(max_pool_out))
        hidden = self.dropout(hidden)  
        return torch.sigmoid(self.fc2(hidden))

accuracy_results = []
precision_results = []
recall_results = []
f1_results = []
mcc_results = []
tnr_results = []

repeats = 5  

for repeat_index in range(repeats):
    print(f"\n========= {repeat_index + 1}/{repeats} ===========")
    model = TCNModel(input_size, num_channels, kernel_size, dropout).to(device)
    criterion = nn.BCELoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    best_test_accuracy = 0.0
    best_model_path_with_suffix = f"{best_model_path}_{k}_{repeat_index}.pth"

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == targets).sum().item()
            train_total += targets.size(0)

        train_accuracy = 100 * train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == targets).sum().item()
                val_total += targets.size(0)

        val_accuracy = 100 * val_correct / val_total

        test_correct, test_total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                predictions = (outputs > 0.5).float()
                test_correct += (predictions == targets).sum().item()
                test_total += targets.size(0)

        test_accuracy = 100 * test_correct / test_total

        scheduler.step()

        combined_accuracy = 0.3 * test_accuracy + 0.7 * val_accuracy
        if combined_accuracy > best_test_accuracy and val_accuracy > 92:
            best_test_accuracy = combined_accuracy
            torch.save(model.state_dict(), best_model_path_with_suffix)
            print(f'Epoch {epoch+1}:  (Test: {test_accuracy:.2f}%, Val: {val_accuracy:.2f}%)')

        elapsed_time = time.time() - start_time
        remaining_time = elapsed_time / (epoch + 1) * (num_epochs - epoch - 1)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, '
              f'Test Acc: {test_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%, '
              f'Time Elapsed: {int(elapsed_time)}s, Time Remaining: {int(remaining_time)}s, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

    model.load_state_dict(torch.load(best_model_path_with_suffix, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions = (outputs > 0.5).float()
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f" {repeat_index + 1} :\n"
          f"Accuracy: {accuracy:.4f}\n"
          f"Precision: {precision:.4f}\n"
          f"Recall: {recall:.4f}\n"
          f"F1 Score: {f1:.4f}\n"
          f"MCC: {mcc:.4f}\n"
          f"TNR: {tnr:.4f}")

    accuracy_results.append(accuracy)
    precision_results.append(precision)
    recall_results.append(recall)
    f1_results.append(f1)
    mcc_results.append(mcc)
    tnr_results.append(tnr)

avg_accuracy = np.mean(accuracy_results)
avg_precision = np.mean(precision_results)
avg_recall = np.mean(recall_results)
avg_f1 = np.mean(f1_results)
avg_mcc = np.mean(mcc_results)
avg_tnr = np.mean(tnr_results)

print("\n=========== mean ===========")
print(f"Accuracy: {avg_accuracy:.4f}")
print(f"Precision: {avg_precision:.4f}")
print(f"Recall: {avg_recall:.4f}")
print(f"F1 Score: {avg_f1:.4f}")
print(f"MCC: {avg_mcc:.4f}")
print(f"TNR: {avg_tnr:.4f}")

