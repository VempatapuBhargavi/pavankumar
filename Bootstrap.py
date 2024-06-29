# Implementing Bootstrapping
import torch
from sklearn.datasets import load_iris
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# Define a simple neural network model for classification
class SimpleNN(torch.nn.Module):
    def _init_(self, input_size, num_classes, hidden_size=16):
        super(SimpleNN, self)._init_()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to train the model
def train_model(model, criterion, optimizer, dataloader):
    model.train()
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Function to evaluate the model
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    return accuracy_score(all_labels, all_preds)

# Bootstrapping
n_bootstraps = 100
boot_accuracies = []

for _ in range(n_bootstraps):
    # Resample the data
    X_resampled, y_resampled = resample(X_tensor.numpy(), y_tensor.numpy(), replace=True, random_state=None)
    X_resampled = torch.from_numpy(X_resampled)
    y_resampled = torch.from_numpy(y_resampled)

    # Identify out-of-bag samples
    oob_indices = list(set(range(len(y_tensor))) - set(np.unique(resample(range(len(y_tensor)), replace=True))))
    X_oob = X_tensor[oob_indices]
    y_oob = y_tensor[oob_indices]

    # Create dataloaders
    train_dataset = TensorDataset(X_resampled, y_resampled)
    oob_dataset = TensorDataset(X_oob, y_oob)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    oob_loader = DataLoader(oob_dataset, batch_size=16, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = SimpleNN(input_size=X_tensor.shape[1], num_classes=len(torch.unique(y_tensor)), hidden_size=16)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, criterion, optimizer, train_loader)

    # Evaluate the model on OOB samples
    oob_accuracy = evaluate_model(model, oob_loader)
    boot_accuracies.append(oob_accuracy)

# Average accuracy and confidence intervals
average_boot_accuracy = np.mean(boot_accuracies)
confidence_interval = (np.percentile(boot_accuracies, 2.5), np.percentile(boot_accuracies, 97.5))

print(f'Bootstrapped average accuracy: {average_boot_accuracy:.4f}')
print(f'95% confidence interval: {confidence_interval}')