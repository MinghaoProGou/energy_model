import subprocess


def run_energy_model_scripts():
    """
    Function to sequentially run multiple Python scripts required for energy model processing.
    It runs:
    1. family_energy_model_trainning.py
    2. housing_energy_consumption.py
    3. housing_floorarea_model.py
    4. housing_age_model.py
    5. SHAP_explain.py
    6. combine_npy.py
    7. SHAP_DT_cluster.py
    """

    scripts = [
        "family_energy_model_trainning.py",
        "housing_energy_consumption.py",
        "housing_floorarea_model.py",
        "housing_age_model.py",
        "SHAP_explain.py",
        "combine_npy.py",
        "SHAP_DT_cluster.py"
    ]

    # Iterate through each script and run them one by one
    for script in scripts:
        try:
            print(f"Running {script}...")
            subprocess.run(['python', script], check=True)
            print(f"{script} completed successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running {script}: {e}")
            break  # Exit if any script fails


if __name__ == "__main__":
    run_energy_model_scripts()
