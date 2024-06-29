# Calculating R-squared
import torch
# Function to calculate R-squared (R²)
def calculate_r2(y_true, y_pred):
    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
    ss_residual = torch.sum((y_true - y_pred) ** 2).item()
    r2 = 1 - (ss_residual / ss_total)
    return r2

# Calculate R-squared
r2 = calculate_r2(y_test, y_pred)
print(f'R-squared (R²): {r2:.4f}')
#calculating R-squared
def calculate_r_squared(yvalues,regression_line):
    y_mean_line = [mean(yvalues) for y in yvalues]
    squared_error_regr = sum((regression_line - yvalues) ** 2)
    squared_error_y_mean = sum((y_mean_line - yvalues) ** 2)
    return 1 - (squared_error_regr / squared_error_y_mean)

#calculate R-squared
regression_line = [(m * x) + b for x in xvalues]
r_squared = calculate_r_squared(yvalues, regression_line)
print(f'R-squared (R2): {r_squared:.4f}')
