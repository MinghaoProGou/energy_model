import os
import numpy as np
from tqdm import tqdm

# Set the directory path containing the .npy files
directory = r'shap_gpu'

# Initialize lists to store the data from different files
data_list = []
shap_list = []
basedata_list = []

# Iterate through the files you want to merge
for i in range(1, 2):  # Adjust the range as needed for other files
    # Define filenames for data, SHAP values, and base values
    filename_data = f'merged_data_simple_{i}.npy'
    filename_base = f'merged_base_simple_{i}.npy'
    filename_shap = f'merged_shap_simple_{i}.npy'

    # Load and append the 'data' files to data_list
    file_path_data = os.path.join(directory, filename_data)
    file_data = np.load(file_path_data)
    data_list.append(file_data)

    # Load and append the 'SHAP values' files to shap_list
    file_path_shap = os.path.join(directory, filename_shap)
    file_shap = np.load(file_path_shap)
    shap_list.append(file_shap)

    # Load and append the 'base values' files to basedata_list
    file_path_base = os.path.join(directory, filename_base)
    file_base = np.load(file_path_base)
    basedata_list.append(file_base)

# Concatenate the data from all files
merged_data_data = np.concatenate(data_list)  # Concatenate along axis 0 (default)
merged_data_shap = np.concatenate(shap_list)
merged_data_base = np.concatenate(basedata_list)

# Print the shape of the concatenated SHAP values for confirmation
print(f"Total SHAP values shape: {merged_data_shap.shape[0]}")

# Save the concatenated data into new .npy files
output_file_data = r'merged_data_simple.npy'
output_file_shap = r'merged_shap_simple.npy'
output_file_base = r'merged_base_simple.npy'

np.save(output_file_data, merged_data_data)
np.save(output_file_shap, merged_data_shap)
np.save(output_file_base, merged_data_base)

# Print confirmation message
print("Merged data saved successfully.")
