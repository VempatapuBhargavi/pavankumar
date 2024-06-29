import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample


# Function to perform bagging and return trained classifiers
def bagging_classifiers(X, y, n_samples=100):
    classifiers = []
    for _ in range(n_samples):
        X_resampled, y_resampled = resample(X, y, random_state=np.random.randint(10000))
        # print(f'Res count : {len(X_resampled)}')
        clf1 = LogisticRegression(max_iter=10000)
        clf2 = DecisionTreeClassifier()
        clf1.fit(X_resampled, y_resampled)
        clf2.fit(X_resampled, y_resampled)
        classifiers.append((clf1, clf2))
    return classifiers


# Train classifiers
classifiers = bagging_classifiers(X_train, y_train, n_samples=10)


# Function to predict using the trained classifiers
def predict_with_classifiers(classifiers, X):
    predictions1 = np.zeros((len(classifiers), X.shape[0]))
    predictions2 = np.zeros((len(classifiers), X.shape[0]))
    for i, (clf1, clf2) in enumerate(classifiers):
        predictions1[i, :] = clf1.predict(X)
        predictions2[i, :] = clf2.predict(X)
    return predictions1, predictions2


# Make predictions on the test set
predictions1, predictions2 = predict_with_classifiers(classifiers, X_test)


# Function to aggregate predictions by voting
def aggregate_predictions(predictions):
    return np.round(np.mean(predictions, axis=0))


# Aggregate predictions for each classifier
aggregated_predictions1 = aggregate_predictions(predictions1)
aggregated_predictions2 = aggregate_predictions(predictions2)
# Combine predictions from both classifiers
combined_predictions = np.round((aggregated_predictions1 + aggregated_predictions2) / 2)


# Function to evaluate the model
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1


# Evaluate individual classifiers
accuracy1, precision1, recall1, f11 = evaluate_model(y_test, aggregated_predictions1)
accuracy2, precision2, recall2, f12 = evaluate_model(y_test, aggregated_predictions2)
# Evaluate combined model
accuracy_combined, precision_combined, recall_combined, f1_combined = evaluate_model(y_test, combined_predictions)
# Print metrics
print("Logistic Regression Bagging:")
print(f"Accuracy: {accuracy1}, Precision: {precision1}, Recall: {recall1}, F1 Score: {f11}")
print("\nDecision Tree Bagging:")
print(f"Accuracy: {accuracy2}, Precision: {precision2}, Recall: {recall2}, F1 Score: {f12}")
print("\nCombined Model:")
print(
    f"Accuracy: {accuracy_combined}, Precision: {precision_combined}, Recall: {recall_combined}, F1 Score: {f1_combined}")
