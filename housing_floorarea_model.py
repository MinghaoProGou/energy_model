# Import necessary libraries
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
import numpy as np
import warnings
import logging
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Setting logging and warning levels
logging.captureWarnings(True)  # Capture and log warnings
logging.getLogger("xgboost").setLevel(logging.WARNING)  # Set XGBoost warnings level to WARNING
warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore FutureWarnings

# 1. Load CSV data into a DataFrame
df = pd.read_csv(r'e_building.csv')

# 2. Feature engineering: calculate 'V' as the product of 'property_area' and 'height'
df['V'] = df['property_area'] * df['height']

# Define features to be used for training and prediction
features = ['property_area', 'height', 'V', 'perimeter', 'DT', 'DA', 'n_uprn']

# Define the target variable
target = 'TA'
X = df[features]
y = df[target]

# Set a random seed for reproducibility and define test set ratio
random_seed = 38
test_size_ratio = 0.5

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=random_seed)


# 3. Define the XGBoost evaluation function for Bayesian optimization
def xgb_evaluate(max_depth, gamma, min_child_weight, reg_alpha, reg_lambda, num_boost_round):
    """
    Evaluation function for XGBoost using the specified hyperparameters.
    Calculates the mean absolute percentage error (MAPE).
    """
    params = {
        'objective': 'reg:squarederror',
        'max_depth': int(max_depth),
        'gamma': gamma,
        'min_child_weight': min_child_weight,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'num_boost_round': int(num_boost_round),
        'device': 'cuda',  # Using GPU for faster training
        'tree_method': 'hist'
    }

    # Create DMatrix for training
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Train the model
    model = xgb.train(params, dtrain)

    # Predict on test data
    y_pred = model.predict(xgb.DMatrix(X_test))

    # Calculate mean absolute percentage error (MAPE)
    mean_error = np.mean(np.abs((y_test - y_pred) / y_test) * 100)

    return -mean_error  # Return negative MAPE for minimization in Bayesian optimization


# 4. Define hyperparameter search space for Bayesian optimization
param_space = {
    'max_depth': (5, 30),
    'gamma': (0, 1),
    'min_child_weight': (0, 15),
    'reg_alpha': (0, 20),
    'reg_lambda': (0, 20),
    'num_boost_round': (100, 2000)
}

# 5. Perform Bayesian Optimization to find the best hyperparameters
optimizer = BayesianOptimization(f=xgb_evaluate, pbounds=param_space, random_state=random_seed)
optimizer.maximize(init_points=10, n_iter=100)

# 6. Retrieve and print the best hyperparameters
best_params = optimizer.max
print("Best hyperparameters:", best_params)

# 7. Train the final model using the best hyperparameters
best_xgb_params = best_params['params']
best_xgb_params['max_depth'] = int(best_xgb_params['max_depth'])  # Ensure max_depth is an integer
best_xgb_params['objective'] = 'reg:squarederror'
best_xgb_params['tree_method'] = 'gpu_hist'

# Create DMatrix for training with the best parameters
dtrain = xgb.DMatrix(X_train, label=y_train)

# Train the final model
model = xgb.train(best_xgb_params, dtrain)

# Save the trained model
model.save_model('TA.json')

# 8. Load the saved model for further predictions
model = xgb.XGBRegressor()
model.load_model('TA.json')

# Predict on the test set
y_pred = model.predict(X_test)

# 9. Calculate evaluation metrics
errors = y_test - y_pred
mae = mean_absolute_error(y_test, y_pred)
mean_error = np.mean(np.abs((y_test - y_pred) / y_test))
percentile_90 = np.percentile(np.abs((y_test - y_pred) / y_test), 90)

print('MAE:', mae)
print('Mean Percentage Error:', mean_error)
print('90% Percentile:', percentile_90)

# 10. Calculate error residuals between the 10th and 90th percentiles
percentile_10 = np.percentile(errors, 2.5)
percentile_90 = np.percentile(errors, 97.5)

values_between_10_and_90 = errors[(errors >= percentile_10) & (errors <= percentile_90)]

# 11. Plot the histogram of error residuals
sns.distplot(values_between_10_and_90, kde=True, bins=20, hist_kws={"alpha": 0.6}, kde_kws={"color": "k", "bw": 0.2})
plt.xlabel('Error Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Error Residuals')

# Display evaluation metrics on the plot
formatted_string = f'MAE: {mae:.2f}\nMean Error: {mean_error * 100:.2f}%\n90% Quartile: {percentile_90 * 100:.2f}%'
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
plt.text((x_max - x_min) * 0.05 + x_min, (y_max - y_min) * 0.8 + y_min, formatted_string, fontsize=9, ha='left')

# Show plot
plt.tight_layout()
plt.show()
