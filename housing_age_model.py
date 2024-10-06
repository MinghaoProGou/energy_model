# Import necessary libraries
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
import numpy as np
import warnings
import logging
from sklearn.metrics import accuracy_score

# Set logging and warning levels
logging.captureWarnings(True)  # Capture and log warnings
logging.getLogger("xgboost").setLevel(logging.WARNING)  # Set XGBoost warnings level to WARNING
warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore FutureWarnings

# 1. Load CSV file into a DataFrame
df = pd.read_csv(r'Integrated data/e_building.csv')

# Drop rows with missing values in the 'DA' column
df.dropna(subset=['DA'], inplace=True)

# Adjust 'DA' values by subtracting 1 (assuming a reclassification or transformation)
df['DA'] = df['DA'] - 1

# Load a pre-trained model for predicting 'TA'
TA_model = xgb.Booster(model_file=r'TA.json')

# Feature engineering: calculate volume 'V' and predict 'TA' using the pre-trained model
df['V'] = df['property_area'] * df['height']
df['TA'] = TA_model.predict(xgb.DMatrix(df[['property_area', 'height', 'V', 'perimeter', 'DT', 'DA']]))

# Define the features and target variable for classification
features = ['property_area', 'height', 'V', 'perimeter', 'DT']
target = 'DA'
X = df[features]
y = df[target]

# Save the processed DataFrame to a CSV file
df.to_csv(r'E:\code\MCMC\data_remaking\eui_building.csv')

# Set a random seed and split the data into training and test sets
s = np.random.randint(1, 100)
test_size = 0.2  # 20% for test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


# Define the objective function for Bayesian optimization
def xgb_cv(max_depth, learning_rate, n_estimators, gamma, min_child_weight, subsample):
    """
    Objective function for Bayesian Optimization.
    This function will train the XGBoost model with the given parameters and return the accuracy.
    """
    params = {
        'objective': 'multi:softmax',  # Multi-class classification
        'num_class': 5,  # Number of classes
        'max_depth': int(max_depth),  # Convert to integer for tree depth
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators),  # Convert to integer for the number of estimators
        'gamma': gamma,
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'device': 'cuda',  # Use GPU for faster training
        'tree_method': 'hist'  # Use histogram-based method for efficiency
    }

    # Initialize and train the model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Predict on the test set and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


# Set the hyperparameter search space for Bayesian Optimization
pbounds = {
    'max_depth': (3, 20),
    'learning_rate': (0.01, 0.5),
    'n_estimators': (50, 500),
    'gamma': (0, 1),
    'min_child_weight': (1, 10),
    'subsample': (0.6, 1.0)
}

# Perform Bayesian Optimization to find the best hyperparameters
optimizer = BayesianOptimization(f=xgb_cv, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=5, n_iter=50)

# Retrieve the best parameters
best_params = optimizer.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])  # Ensure n_estimators is an integer
best_params['max_depth'] = int(best_params['max_depth'])  # Ensure max_depth is an integer
best_params['device'] = 'cuda'  # Use GPU for final model
best_params['tree_method'] = 'hist'

# Initialize XGBoost classifier with the best hyperparameters
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=5,  # Number of classes for multi-class classification
    **best_params
)

# Train the final model using the best hyperparameters
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the final model to a file
model.save_model('DA.json')
