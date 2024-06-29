
# ROC Curve and AUC
import torch
import numpy as np
import matplotlib as plt
import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc
# Calculate probabilities
model.eval()
y_prob = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        y_prob.extend(probabilities.numpy())
y_prob = np.array(y_prob)  # Convert list of NumPy arrays to a single NumPy array
y_prob = torch.from_numpy(y_prob)  # Convert the NumPy array to a PyTorch tensor
# Binarize the output for ROC
y_test_binarized = torch.nn.functional.one_hot(y_test).numpy()
y_prob = y_prob.numpy()
# Plot ROC curve and calculate AUC for each class
plt.figure(figsize=(10, 8))
for i in range(len(data.target_names)):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {data.target_names[i]} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
