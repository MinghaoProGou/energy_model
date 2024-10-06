import os
import pandas as pd
import xgboost as xgb
import json
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

def _auto_cohorts(data, values, base, feature_names, max_cohorts):
    """
    This function uses a DecisionTreeRegressor to group data into cohorts based on SHAP values.
    It fits a decision tree that separates the SHAP values into distinct cohorts,
    and it labels each cohort with a unique decision path.
    """

    # Fit a decision tree regressor to create cohorts
    tree_model = DecisionTreeRegressor(max_leaf_nodes=max_cohorts)
    tree_model.fit(data, values)

    # Get the decision paths for each data point
    paths = tree_model.decision_path(data).toarray()
    path_names = []
    connections = {}

    # Iterate through the data to create path names and connections for each cohort
    for i in tqdm(range(data.shape[0])):
        connection = {'feature_name': [], 'calculate': [], 'threshold': []}
        for j in range(len(paths[i])):
            if paths[i, j] > 0:
                feature = tree_model.tree_.feature[j]
                threshold = tree_model.tree_.threshold[j]
                feature_name = feature_names[feature]
                val = data[i, feature]

                # Determine whether the value is less than or greater than the threshold
                if val < threshold:
                    if feature_name in connection['feature_name']:
                        index1 = [index for index, value in enumerate(connection['feature_name']) if feature_name in value]
                        ca = [connection['calculate'][ix] for ix in index1]
                        if '<' in ca:
                            index2 = [index for index, value in enumerate(ca) if '<' in value][0]
                            idfn = index1[index2]
                            del connection['feature_name'][idfn]
                            del connection['calculate'][idfn]
                            del connection['threshold'][idfn]
                    connection['feature_name'].append(feature_name)
                    connection['calculate'].append('<')
                    connection['threshold'].append(threshold)
                else:
                    if feature_name in connection['feature_name']:
                        index1 = [index for index, value in enumerate(connection['feature_name']) if feature_name in value]
                        ca = [connection['calculate'][ix] for ix in index1]
                        if '>=' in ca:
                            index2 = [index for index, value in enumerate(ca) if '>=' in value][0]
                            idfn = index1[index2]
                            del connection['feature_name'][idfn]
                            del connection['calculate'][idfn]
                            del connection['threshold'][idfn]
                    connection['feature_name'].append(feature_name)
                    connection['calculate'].append('>=')
                    connection['threshold'].append(threshold)

        # Create a path name based on the features and thresholds
        name = " & ".join([f"{connection['feature_name'][i]}{connection['calculate'][i]}{connection['threshold'][i]:.2f}" for i in range(len(connection['feature_name']))])
        path_names.append(name)  # Append the path name
        connections[name] = connection

    # Convert path names to numpy array
    path_names = np.array(path_names)

    # Split the data into cohorts based on unique path names
    cohorts = {}
    for name in np.unique(path_names):
        cohort_data = data[path_names == name]
        cohorts[name] = {
            'data': cohort_data.tolist(),
            'base': np.mean(base).tolist(),
            'feature_names': feature_names
        }
        connections[name]['percent'] = cohort_data.shape[0] / data.shape[0]

    return cohorts, connections

# Load the SHAP values, base values, and feature names
values = np.load(r'merged_shap_simple.npy')
base = np.load(r'merged_base_simple.npy')
feature_names = ['property_area', 'height', 'V', 'DA', 'DT', 'perimeter']

# Load the dataset containing the necessary features
df = pd.read_csv(r'eui_building.csv')
data = df[feature_names].values

# Create directories to save the JSON output
output_dir = r'json_simple3'
os.makedirs(output_dir, exist_ok=True)

# Generate cohorts and save connections to JSON files for each iteration
for i in range(3, 50):  # Adjust the range as needed
    print(f"Processing with {i} cohorts...")
    cohorts, connections = _auto_cohorts(data, values, base, feature_names, i)

    # Save the connections dictionary as a JSON file
    with open(os.path.join(output_dir, f'{i}_cluster.json'), "w") as json_file:
        json.dump(connections, json_file)

    print(f"Saved: {i}_cluster.json")

print("Cohorting and saving complete.")
