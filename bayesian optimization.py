import torch
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skopt.utils import use_named_args
from torch.utils.data import DataLoader, TensorDataset

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)


# Model and Optimization Process
# Define a simple neural network model for classification
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=16):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


from skopt.space import Integer, Real

# Define the search space for Bayesian optimization
bayesian_space = [
    Real(0.0001, 0.01, prior='log-uniform', name='lr'),
    Integer(16, 64, name='batch_size'),
    Integer(50, 150, name='epochs')
]


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


# Implementing Bayesian Optimization
from skopt import gp_minimize


# Define the objective function for Bayesian optimization
@use_named_args(bayesian_space)
def objective(lr, batch_size, epochs):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=int(batch_size), shuffle=False)

    model = SimpleNN(input_size=X_tensor.shape[1], num_classes=len(torch.unique(y_tensor)))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    for epoch in range(epochs):
        train_model(model, criterion, optimizer, train_loader)

    # Evaluate the model
    accuracy = evaluate_model(model, test_loader)

    # Return the negative accuracy (since we minimize the objective function)
    return -accuracy


# Perform Bayesian optimization
result = gp_minimize(objective, bayesian_space, n_calls=20, random_state=42)

best_params = {
    'lr': result.x[0],
    'batch_size': result.x[1],
    'epochs': result.x[2]
}

print(f'Best parameters: {best_params}')
print(f'Best accuracy: {-result.fun:.4f}')

