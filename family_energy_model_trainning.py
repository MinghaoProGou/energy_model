# Import necessary libraries
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import seaborn as sns
import warnings
import logging
import datetime

# Set logging level to suppress warnings and unnecessary output from XGBoost
logging.captureWarnings(True)
logging.getLogger("xgboost").setLevel(logging.WARNING)

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load dataset
file_path = 'Integrated data/energy_use_london.csv'
df = pd.read_csv(file_path)

# Define features and target variable
features = ['DWage', 'DWtype', 'FloorArea', 'sap12']
print(features)

# Define target and weights
w = df['aagph1718']  # Sample weight
y = df['e_total']  # Target variable

# Split dataset into training and testing sets
s = 38  # Seed for reproducibility
ts = 0.7  # Test set ratio

X_train, X_test, y_train, y_test = train_test_split(df[features], y, test_size=ts, random_state=s)
w_train, w_test = train_test_split(w, test_size=ts, random_state=s)


# Define the evaluation function for Bayesian optimization
def xgb_evaluate(max_depth, min_child_weight, gamma, colsample_bytree, reg_alpha, reg_lambda, num_boost_round):
    """
    Evaluate the XGBoost model with a given set of hyperparameters.
    This function is used by Bayesian optimization to minimize the MAE.
    """
    params = {
        'objective': 'reg:squarederror',
        'max_depth': int(round(max_depth)),
        'min_child_weight': int(round(min_child_weight)),
        'gamma': gamma,
        'colsample_bytree': colsample_bytree,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'eval_metric': 'mae',
        'num_boost_round': int(round(num_boost_round))
    }

    reg = xgb.XGBRegressor(**params)
    model = reg.fit(X_train, y_train, sample_weight=w_train)

    # Make predictions and compute MAE
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return -mae  # Negative MAE because Bayesian optimization minimizes the objective function


# Perform Bayesian optimization to find the best hyperparameters
xgb_bo = BayesianOptimization(xgb_evaluate, {
    'max_depth': (0, 30),
    'min_child_weight': (0, 15),
    'gamma': (0, 1),
    'colsample_bytree': (0.5, 1),
    'reg_alpha': (0, 20),
    'reg_lambda': (0, 20),
    'num_boost_round': (100, 2000)
})

# Maximize the optimization process with initial points and iterations
xgb_bo.maximize(init_points=20, n_iter=200, acq='ucb')

# Extract the best parameters
best_params = xgb_bo.max['params']
best_params['max_depth'] = int(round(best_params['max_depth']))
best_params['min_child_weight'] = int(round(best_params['min_child_weight']))
best_params['num_boost_round'] = int(round(best_params['num_boost_round']))

# Convert data to DMatrix format for XGBoost training
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# Train the final model using the best parameters
model = xgb.train(best_params, dtrain)
model.save_model('uprn_model.model')

# Save the model with a time-based filename
current_time = datetime.datetime.now()
model_name = f'uprn_{current_time.hour}_{current_time.minute}.model'
model.save_model(model_name)

# Predict on the test data
y_pred = model.predict(dtest)

# Calculate errors and residuals
mae = mean_absolute_error(y_test, y_pred)
errors = np.abs((y_test - y_pred) * w_test)
percentile_90 = np.percentile(errors, 90)
mean_error = np.mean(errors)

# Print evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("90% Percentile:", percentile_90)

# Calculate skewness and kurtosis
skewness = stats.skew(errors)
kurtosis = stats.kurtosis(errors)
print("Skewness:", skewness)
print("Kurtosis:", kurtosis)

# Plot the distribution of errors
sns.distplot(errors, kde=True, bins=50, hist_kws={"alpha": 0.6}, kde_kws={"color": "k", "bw": 0.4})
plt.xlabel('Error Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Error Residuals')

# Add evaluation metrics to the plot
formatted_string = f'MAE: {mae:.2f}\n90% Quartile: {percentile_90:.2f}%'
plt.text(0.05, 0.95, formatted_string, ha='left', va='top', transform=plt.gca().transAxes)

# Save the plot
plt.tight_layout()
plt.savefig(r'uprn_energy_model.png')

