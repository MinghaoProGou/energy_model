# Import necessary libraries
import pandas as pd
import xgboost as xgb
import time
import datetime

# Load pre-trained XGBoost model
model = xgb.XGBRegressor()
model.load_model(r'uprn_model.model')

# Load the dataset for prediction
file_path = r'Integrated data/london_building11.csv'
df = pd.read_csv(file_path)

# Make a copy of the original DataFrame for renaming and transformation
df1 = df.copy()

# Load additional data (for merging later)
ldf = pd.read_csv(r'l_result.csv')

# Rename columns to match the model's feature names
df1.rename(columns={
    'DT': 'DWtype',
    'DA': 'DWage',
    'FL': 'FloorArea',
    'AGG_EPC_CURRENT_ENERGY_EFFICIENCY_BY_FLOORAREA': 'sap12'
}, inplace=True)

# Define features used in prediction
features = ['DWage', 'DWtype', 'FloorArea', 'sap12']
X = df1[features]

# Predict using the pre-trained model
t1 = time.time()
e = model.predict(X)
print(f"Prediction time: {time.time() - t1:.2f} seconds")

# Add the prediction results to the original dataframe
df['e_total'] = e

# Ensure 'height' is numeric (coerce errors)
df['height'] = pd.to_numeric(df['height'], errors='coerce')

# Select relevant columns for LSOA and MSOA
lsoa = df[['toid', 'LSOA', 'MSOA']]

# Calculate the number of households by counting 'toid'
res = df['toid'].value_counts().reset_index()
res.columns = ['toid', 'n_uprn']

# Aggregate features by 'toid' (group by) and calculate necessary statistics
e_total = df.groupby('toid')['e_total'].sum().reset_index()
DWtype = df.groupby('toid')['DT'].apply(lambda x: x.mode()[0]).reset_index()  # Mode for categorical features
DWage = df.groupby('toid')['DA'].apply(lambda x: x.mode()[0]).reset_index()
FloorArea = df.groupby('toid')['TA'].sum().reset_index()
Footprint_area = df.groupby('toid')['property_area'].mean().reset_index()
FA = df.groupby('toid')['FA'].mean().reset_index()
height = df.groupby('toid')['height'].mean().reset_index()

# Merge all aggregated data into one dataframe 'res'
for i in [lsoa, e_total, DWtype, DWage, FloorArea, Footprint_area, height, FA, ldf]:
    res = pd.merge(res, i, on='toid', how='left')

# Select relevant columns for the final result
res = res[['toid', 'LSOA', 'MSOA', 'n_uprn', 'e_total', 'DT', 'DA', 'TA', 'FA', 'property_area', 'height', 'perimeter']]

# Remove duplicates based on 'toid'
res = res.drop_duplicates(subset='toid')

# Print total processing time
print(f"Total time elapsed: {time.time() - t1:.2f} seconds")

# Save the result to a CSV file
res.to_csv(r'Integrated data/e_building.csv', index=False)

# Get the current time to use in the file name
current_time = datetime.datetime.now()
current_hour = current_time.hour
current_minute = current_time.minute

# Save the result to a new CSV file with the current hour and minute in the filename
res.to_csv(fr'E:Integrated data/e_building_{current_hour}_{current_minute}.csv', index=False)
