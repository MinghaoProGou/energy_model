# Import necessary libraries
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
import numpy as np
import warnings
import logging
import shap
import pickle
from sklearn.metrics import mean_absolute_error
import datetime

# Setting logging and warning levels
logging.captureWarnings(True)  # Capture warnings
logging.getLogger("xgboost").setLevel(logging.WARNING)  # Set XGBoost's warning level to WARNING
warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore FutureWarnings

# Load pre-trained model for DA prediction
DA_model = xgb.Booster(model_file=r'DA.json')

# 1. Load dataset from CSV
df = pd.read_csv(r'Integrated data/e_building.csv')

# Feature engineering: calculate volume 'V' and identify rows where DA is missing (0)
df['V'] = df['property_area'] * df['height']
nan_rows = df[df['DA'] == 0]
X_nan = nan_rows[['property_area', 'height', 'V', 'perimeter', 'DT']]

# Predict missing 'DA' values using pre-trained DA model
if not X_nan.empty:
    dtest = xgb.DMatrix(X_nan)
    predictions = DA_model.predict(dtest)
    df.loc[nan_rows.index, 'DA'] = predictions

# Load pre-trained model for TA prediction
TA_model = xgb.Booster(model_file=r'E:\code\MCMC\data_remaking\TA3.json')

# Feature engineering: calculate ratio 'r'
df['r'] = df['V'] / (df['perimeter'] * df['height'] + df['property_area'])

# Predict 'TA' using the pre-trained TA model
df['TA'] = TA_model.predict(xgb.DMatrix(df[['property_area', 'height', 'V', 'perimeter', 'DT', 'DA', 'n_uprn']]))

# Calculate Energy Use Intensity (EUI)
df['EUI'] = df['e_total'] / df['TA']

# Define features and target for the model
features = ['property_area', 'height', 'V', 'DA', 'DT', 'perimeter', 'r']
target = 'EUI'
X = df[features]
y = df[target]

# Save the processed dataset to a CSV file
df.to_csv(r'Integrated data/eui_building.csv')

# Split data into training and testing sets
s = np.random.randint(1, 100)  # Random seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=s)

# 2. Define the XGBoost model's objective function for Bayesian optimization
def xgb_evaluate(max_depth, gamma, min_child_weight, reg_alpha, reg_lambda, num_boost_round):
    """
    Function to evaluate the model's performance and return negative mean absolute error (MAE).
    This function will be used by Bayesian Optimization.
    """
    params = {
        'objective': 'reg:squarederror',
        'max_depth': int(max_depth),
        'gamma': gamma,
        'min_child_weight': min_child_weight,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'num_boost_round': int(num_boost_round),
        'device': 'cuda',  # Use GPU for faster training
        'tree_method': 'hist'  # Efficient tree construction
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(params, dtrain)
    y_pred = model.predict(xgb.DMatrix(X_test))
    mae = mean_absolute_error(y_test, y_pred)  # Calculate MAE

    return -mae  # Return negative MAE for minimization

# 3. Define the hyperparameter search space
param_space = {
    'max_depth': (10, 15),
    'gamma': (0, 1),
    'min_child_weight': (0, 2),
    'reg_alpha': (0, 20),
    'reg_lambda': (0, 20),
    'num_boost_round': (100, 2000)
}

# 4. Use Bayesian Optimization to find the best hyperparameters
optimizer = BayesianOptimization(f=xgb_evaluate, pbounds=param_space, random_state=s)
optimizer.maximize(init_points=10, n_iter=100)

# 5. Retrieve the best parameters
best_params = optimizer.max
print("Best hyperparameters:", best_params)

# 6. Train the final model using the best hyperparameters
best_xgb_params = best_params['params']
best_xgb_params['max_depth'] = int(best_xgb_params['max_depth'])
best_xgb_params['objective'] = 'reg:squarederror'
best_xgb_params['tree_method'] = 'gpu_hist'

# Train the final model
dtrain = xgb.DMatrix(X_train, label=y_train)
model = xgb.train(best_xgb_params, dtrain)

# Save the final model
model.save_model('EUI.json')

# 7. Evaluate the model on the entire dataset
y_pred = model.predict(xgb.DMatrix(X))

# Calculate performance metrics
mae = mean_absolute_error(y, y_pred)
mean_percent_error = np.mean(np.abs((y - y_pred) / y) * 100)
percentile_90 = np.percentile(np.abs((y - y_pred) / y) * 100, 90)

print(f"MAE: {mae}")
print(f"Mean Percentage Error: {mean_percent_error}")
print(f"90% Percentile: {percentile_90}")

# 8. Save SHAP explainer using GPU
model.set_param({"predictor": "gpu_predictor"})
explainer_gpu = shap.TreeExplainer(model=model)

# Save the SHAP explainer object to a file
explainer_file_path = r'EUI_explainer_gpu_simple.pkl'
with open(explainer_file_path, 'wb') as f:
    pickle.dump(explainer_gpu, f)

# Save with a timestamped filename
current_time = datetime.datetime.now()
current_hour = current_time.hour
current_minute = current_time.minute
explainer_timestamped_file = rf'EUI_explainer_gpu_simple_{current_hour}_{current_minute}.pkl'

# Save the SHAP explainer with a timestamped name
with open(explainer_timestamped_file, 'wb') as f:
    pickle.dump(explainer_gpu, f)

# Output the current time for reference
print(f"Current time: {current_hour}:{current_minute}")
