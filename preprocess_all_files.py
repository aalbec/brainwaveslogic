import os
import pandas as pd
from glob import glob
from preprocess_module import preprocess_file, motor_imagery_markers

# Directory containing all 30 EEG files
data_directory = "eeg_muse2_moto_imagery_brain_ electrical_activity/"
output_file = "preprocessed_eeg_combined_data.csv"

# Get all CSV file paths in the directory
file_paths = glob(os.path.join(data_directory, "*.csv"))

# Initialize an empty list to store processed DataFrames
all_processed_data = []

# Iterate through each file and preprocess it
for file_index, file_path in enumerate(file_paths, start=1):
    file_id = f"file_{file_index:02d}"  # Generate unique File_ID
    print(f"Processing {file_path} as {file_id}...")

    # Preprocess the file
    processed_data = preprocess_file(file_path, file_id, motor_imagery_markers)

    # Append the processed DataFrame to the list
    all_processed_data.append(processed_data)

# Combine all processed files into a single DataFrame
combined_data = pd.concat(all_processed_data, ignore_index=True)

# Save the combined DataFrame to a CSV file
combined_data.to_csv(output_file, index=False)

print(f"All files processed and combined. Output saved to {output_file}.")
