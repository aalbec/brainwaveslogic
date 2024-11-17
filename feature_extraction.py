import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from preprocess_module import motor_imagery_markers

file_path = "preprocessed_eeg_combined_data.csv"  # Update with the correct file path
eeg_data = pd.read_csv(file_path)

# Display basic info and the first few rows to verify successful loading
eeg_data_info = eeg_data.info()
eeg_data_head = eeg_data.head()
print("\n=== EEG Data Info ===")
print(eeg_data_info)
print("\n=== First 5 Rows of EEG Data ===")
print(eeg_data_head)

event_features = eeg_data.groupby(["Event_Count", "Task_Label", "File_ID"]).agg(
    {
        "Delta_TP9": ["mean", "var", "min", "max"],
        "Delta_AF7": ["mean", "var", "min", "max"],
        "Delta_AF8": ["mean", "var", "min", "max"],
        "Delta_TP10": ["mean", "var", "min", "max"],
        "Theta_TP9": ["mean", "var", "min", "max"],
        "Theta_AF7": ["mean", "var", "min", "max"],
        "Theta_AF8": ["mean", "var", "min", "max"],
        "Theta_TP10": ["mean", "var", "min", "max"],
        "Alpha_TP9": ["mean", "var", "min", "max"],
        "Alpha_AF7": ["mean", "var", "min", "max"],
        "Alpha_AF8": ["mean", "var", "min", "max"],
        "Alpha_TP10": ["mean", "var", "min", "max"],
        "Beta_TP9": ["mean", "var", "min", "max"],
        "Beta_AF7": ["mean", "var", "min", "max"],
        "Beta_AF8": ["mean", "var", "min", "max"],
        "Beta_TP10": ["mean", "var", "min", "max"],
        "Gamma_TP9": ["mean", "var", "min", "max"],
        "Gamma_AF7": ["mean", "var", "min", "max"],
        "Gamma_AF8": ["mean", "var", "min", "max"],
        "Gamma_TP10": ["mean", "var", "min", "max"],
    }
)

# Flatten MultiIndex columns
event_features.columns = ["_".join(col) for col in event_features.columns]

# Reset the index to retain 'Event_Count', 'Task_Label', and 'File_ID'
event_features = event_features.reset_index()

print("\n=== Aggregated Event Features ===")
print(event_features.head())

# Identify columns to scale (exclude 'Event_Count', 'Task_Label', and 'File_ID')
identifier_columns = ["Event_Count", "Task_Label", "File_ID"]
feature_columns = [
    col for col in event_features.columns if col not in identifier_columns
]

# Separate features and identifiers
features = event_features[feature_columns]
identifiers = event_features[identifier_columns]

# Standardization
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

# Normalization
normalizer = MinMaxScaler()
normalized_features = normalizer.fit_transform(standardized_features)

# Create DataFrame for standardized and normalized features
normalized_features_df = pd.DataFrame(normalized_features, columns=feature_columns)

# Combine identifiers with normalized features
final_scaled_features = pd.concat(
    [identifiers.reset_index(drop=True), normalized_features_df], axis=1
)

print("\n=== Final Scaled and Normalized Features ===")
print(final_scaled_features.head())

# Save the resulting DataFrame
output_file = "extracted_normalized_features_combined.csv"
final_scaled_features.to_csv(output_file, index=False)
print(f"\nFeature extraction complete. Data saved to {output_file}.")

##################### Sanitary check ###########
# Load the combined dataset
combined_data = pd.read_csv("extracted_normalized_features_combined.csv")

# Step 1: Check for missing values
nan_summary = combined_data.isnull().sum()
print("Missing Values Summary:")
print(nan_summary)

# Step 2: Verify unique Event_Count within each File_ID
duplicate_events = combined_data.groupby("File_ID")["Event_Count"].apply(
    lambda x: x.duplicated().sum()
)
print("\nDuplicate Event_Count within each File_ID:")
print(duplicate_events)

# Step 3: Verify consistent number of samples per event
samples_per_event = combined_data.groupby("Event_Count").size().unique()
print("\nSamples per Event (should be consistent):")
print(samples_per_event)

# Step 4: Check the total number of unique Event_Counts
unique_event_counts = combined_data["Event_Count"].nunique()
print(f"\nTotal Unique Event_Counts: {unique_event_counts}")

# Step 5: Verify Task_Label consistency
unexpected_labels = combined_data.loc[
    ~combined_data["Task_Label"].isin(motor_imagery_markers.values()), "Task_Label"
].unique()
print("\nUnexpected Task_Labels (should be empty):")
print(unexpected_labels)

# Step 6: File ID to Event Count mapping
file_event_mapping = combined_data.groupby("File_ID")["Event_Count"].nunique()
print("\nNumber of Events per File:")
print(file_event_mapping)

# Step 7: Dataset shape and preview
print("\nDataset Shape:", combined_data.shape)
print("\nSample Rows:")
print(combined_data.head())
