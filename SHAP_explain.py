# Import necessary libraries
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import xgboost as xgb
import os

# Load the SHAP explainer object from a pickle file
with open(r'EUI_explainer_gpu_simple.pkl', 'rb') as f:
    loaded_explainer = pickle.load(f)

# Assign the loaded explainer to a variable for use
explainer = loaded_explainer
print('SHAP Explainer Loaded Successfully')

# Load the dataset
df = pd.read_csv(r'E:\code\MCMC\data_remaking\eui_building.csv')

# Feature engineering: calculate volume 'V'
df['V'] = df['property_area'] * df['height']

# Load the pre-trained TA model and predict 'TA'
TA_model = xgb.Booster(model_file=r'E:\code\MCMC\building_energy\TA3.json')
df['TA'] = TA_model.predict(xgb.DMatrix(df[['property_area', 'height', 'V', 'perimeter', 'DT', 'DA', 'n_uprn']]))

# Print status to indicate data preparation is complete
print('Data Preparation Complete')

# Set the chunk size for processing in batches
am = 1  # Chunk multiplier
nn = df.shape[0] // am  # Total number of rows divided by chunk size
print(f"Number of rows per chunk: {nn}")

# Define the feature columns for SHAP value calculation
features = ['property_area', 'height', 'V', 'DA', 'DT', 'perimeter', 'r']
Xn = df[features]  # Subset the DataFrame to include only the features
os.makedirs(r'shap_gpu', exist_ok=True)
# Loop through the data in chunks and compute SHAP values
for n in tqdm(range(1, 2)):  # Adjust range as needed for more chunks
    # Define the chunk of data to process
    if (n + 1) * nn < df.shape[0]:
        X = Xn.loc[(n - 1) * nn:n * nn]
    else:
        X = Xn.loc[(n - 1) * nn:df.shape[0]]

    print(f"Processing chunk {n} with {X.shape[0]} rows")

    # Calculate SHAP values using the explainer
    shap_values = explainer(X)

    # Extract different components of SHAP values (data, base values, SHAP values)
    data = shap_values.data  # The feature values
    base_values = shap_values.base_values  # The base values
    shap_values_data = shap_values.values  # The SHAP values

    # Save the SHAP results into separate numpy files for later use
    np.save(fr'shap_gpu\merged_data_simple_{n}.npy', data)
    np.save(fr'shap_gpu\merged_base_simple_{n}.npy', base_values)
    np.save(fr'shap_gpu\merged_shap_simple_{n}.npy', shap_values_data)

# Print completion status
print("SHAP values calculated and saved successfully.")
