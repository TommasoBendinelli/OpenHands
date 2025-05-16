import mlcroissant as mlc

import os
import json
import hashlib
from datetime import datetime, timezone
import mlcroissant as mlc # The official mlcroissant library
import pandas as pd

subdataset_descriptions = {
    "channel_corr": "Binary classification dataset distinguishing between correlated and uncorrelated time series channels. Label 0 indicates channels share an underlying common signal component (correlated), while label 1 indicates independent, uncorrelated channels. Each sample contains two channels with random scales and offsets, requiring correlation detection regardless of linear transformations.",
    
    "channel_divergence": "Dataset for detecting whether two time series channels diverge over time (label 1) or maintain a bounded gap (label 0). Tests a model's ability to identify when channels fail to maintain a consistent relationship over time.",
    
    "cofounded_group_outlier": "Binary classification with a confounding variable present only in training. The true rule is group-level: label 1 if outlier ratio ≥ 0.08 (where outlier = |signal| > 3). In training data, color perfectly correlates with labels ('red' = label 1, 'blue' = label 0), but this shortcut is absent in test data, forcing models to learn the true group-level outlier pattern.",
    
    "common_frequency": "Dataset for identifying whether two signals share a common frequency component (label 0) or have different dominant frequencies (label 1). Challenges models to identify frequency relationships between signals regardless of phase differences and noise.",
    
    "dominant_feature": "Binary classification dataset where label 1 indicates one feature dominates the others in magnitude. The task requires identifying when a particular feature's value significantly outweighs others, testing a model's ability to detect relative importance among features.",
    
    "find_peaks": "Peak-count benchmark with Gaussian-shaped, non-overlapping peaks. Class 0 has 3-4 peaks while class 1 has 5-6 peaks, all satisfying minimum amplitude and area constraints. The decision boundary 'count ≥ 4.5' perfectly separates the classes, testing a model's ability to count significant events in noisy signals.",
    
    "ground_mean_threashold": "Group-based classification where a group is labeled 1 if mean(value_0) > 3.0. Only some rows (~60%) in positive-label groups carry signal (drawn from N(6,1)), while the rest contain noise. Distractor columns contain pure noise, forcing models to use group-level aggregation rather than row-level shortcuts.",
    
    "outlier_ratio": "Dataset for detecting groups with high outlier ratios. Groups are labeled 1 if their outlier ratio (proportion of values where |signal| > 3) exceeds 0.08. Requires models to perform group-level aggregation to identify anomalous patterns.",
    
    "periodic_presence": "Time series classification between periodic signals (label 0: sine-wave + noise) and aperiodic signals (label 1: colored noise). Each signal is normalized to unit variance to prevent simple variance-based shortcuts, requiring spectral analysis to detect the presence of dominant frequency components.",
    
    "predict_ts_stationarity": "Dataset for predicting whether a time series is stationary (label 0) or non-stationary (label 1) based on statistical properties. Tests a model's ability to identify when a signal's statistical properties remain constant over time.",
    
    "row_max_abs": "Binary classification dataset where label 1 corresponds to rows where max(|feat1…feat12|) > 4.0. Tests a model's ability to identify when any feature in a row exceeds a specific magnitude threshold.",
    
    "row_variance": "Dataset distinguishing between high variance (label 1) and low variance (label 0) feature distributions within each row. Challenges models to identify relative dispersion of values across features.",
    
    "sign_rotated_generator": "Classification task based on the relationship between feature signs. The decision boundary relates to whether features maintain consistent quadrant relationships, with a margin threshold determining classification. Tests a model's ability to detect sign-based patterns across features.",
    
    "simultanus_spike": "Dataset for detecting whether spikes in two channels occur simultaneously (label 0) or independently (label 1). Each sample contains two time series with baseline offsets and random spike locations, testing a model's ability to identify temporal coincidences.",
    
    "sum_threshold": "Classification task based on whether the sum of specific features exceeds a threshold. Label 1 indicates sum(feat1,feat2,feat3) > 1.5. Tests a model's ability to combine information across features and apply thresholds.",
    
    "variance_burst": "Time series classification dataset for detecting localized high-variance episodes (bursts) in signals. Label 1 indicates the presence of a short segment with significantly higher variance (5x the baseline), while label 0 represents consistent variance throughout. The dataset challenges models to detect temporary volatility increases in signals regardless of their baseline level or offset, requiring sensitivity to relative variance changes rather than absolute magnitude.",

    "zero_crossing": "Time series dataset for detecting the presence of a specific frequency band. Label 1 indicates signals containing components in a target frequency band (typically 5-10 Hz), while label 0 indicates signals without those components. Tests a model's ability to identify specific frequency characteristics."
}

def get_sha256(file_path):
    """Computes the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def create_croissant_dataset_with_library(base_path, output_json_file):
    """
    Generates a Croissant JSON-LD file using the mlcroissant library.

    Args:
        base_path (str): The root path containing subtask folders.
        output_json_file (str): The path where the Croissant JSON-LD will be saved.
    """
    # --- Customize these values for your dataset ---
    dataset_name = "EDAx17"
    dataset_description = """
    A comprehensive collection of datasets for explorative data analysis benchmarks, designed to test and evaluate machine learning models on common pattern recognition tasks. This collection contains 17 different datasets, each focusing on a specific analytical challenge commonly encountered in data science.

    The datasets span various domains including time series analysis, signal processing, statistical pattern recognition, and group-based anomaly detection. Each dataset is constructed with carefully designed patterns that models should learn to identify, with clear binary classification targets (label 0 or 1) that represent success in recognizing these patterns.

    These datasets are particularly valuable for benchmarking models' ability to discover meaningful patterns in data without relying on shortcuts or spurious correlations. Many datasets include deliberate challenges like confounding variables in training that are absent in testing, forcing models to learn the true underlying patterns rather than superficial correlations.

    All datasets include both training and test splits with consistent labeling, making them suitable for supervised learning evaluations and benchmarking.
    """
    # This should be a unique, persistent URL identifying your dataset
    dataset_url = "https://anonymous.4open.science/r/ExplorationDataAnalysis-EDx17/evaluation/benchmarks/error_bench/tasks"
    creator_name = "Tommaso Bendinelli"
    creator_email = "tommaben@ethz.ch"
    # Use SPDX identifiers (e.g., "apache-2.0", "mit") or a URL to a custom license
    dataset_license_str = "cc-by-nc-4.0"
    dataset_version = "1.0.0"
    # --- End of customization ---

    # Initialize Croissant Metadata object
    # This object will hold all information about the dataset.
    metadata = mlc.Metadata(
        name=dataset_name,
        description=dataset_description,
        url=dataset_url, # Canonical URL of the dataset itself
        license=dataset_license_str,
        version=dataset_version,
        # date_published=datetime.now(timezone.utc),
        # You can add multiple creators (Person or Organization)
        creators=[mlc.Person(name=creator_name, email=creator_email)]
    )

    # Ensure base_path is absolute for reliable relative path calculations
    base_path = os.path.abspath(base_path)
    # The directory where the output Croissant JSON file will be saved
    output_json_dir = os.path.dirname(os.path.abspath(output_json_file))

    subtask_folders = sorted([
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    ])

    if not subtask_folders:
        print(f"Warning: No subtask folders found in {base_path}")
        return

    print(f"Found {len(subtask_folders)} subtask folders: {subtask_folders}")

    for subtask_name in subtask_folders:
        subtask_dir_path = os.path.join(base_path, subtask_name)
        print(f"\nProcessing subtask: {subtask_name}")

        # Define the set of files expected in each subtask folder
        files_info = [
            {"name_suffix": "train_data", "filename": "train.csv", "description_prefix": "Training data"},
            {"name_suffix": "train_labels", "filename": "train_labels.csv", "description_prefix": "Training labels"},
            {"name_suffix": "test_data", "filename": "test.csv", "description_prefix": "Test data"},
            {"name_suffix": "test_ground_truth", "filename": "test_gt.csv", "description_prefix": "Test ground truth"},
        ]

        for file_info in files_info:
            file_path = os.path.join(subtask_dir_path, file_info["filename"])

            if not os.path.exists(file_path):
                print(f"  WARNING: File not found, skipping: {file_path}")
                continue

            file_sha256 = get_sha256(file_path)
            
            # content_url should be relative to the location of the Croissant JSON file
            content_url = os.path.relpath(file_path, start=output_json_dir)
            # Ensure POSIX-style paths for URLs, as Croissant expects
            content_url = content_url.replace(os.sep, '/')

            print(f"  Found: {file_info['filename']}, SHA256: {file_sha256}, contentUrl: {content_url}")

            file_object_id = f"{subtask_name}-{file_info['name_suffix'].replace('_', '-')}"
            file_object_name = f"{subtask_name}_{file_info['name_suffix']}"

            # 1. Create FileObject for the current CSV file
            file_object = mlc.FileObject(
                id=file_object_id, # Unique ID for this FileObject within the dataset
                name=file_object_name,
                description=f"{file_info['description_prefix']} for subtask {subtask_name}.",
                content_url=content_url,
                encoding_formats="text/csv",
                sha256=file_sha256,
            )
            metadata.distribution.append(file_object) # Add to dataset's distribution list

            # 2. Create RecordSet to describe the data within this FileObject
            record_set_id = f"records-{file_object_id}"
            record_set_name = f"{file_object_name}_records"
            
            record_set = mlc.RecordSet(
                id=record_set_id, # Unique ID for this RecordSet
                name=record_set_name,
                description=f"Records from {file_info['description_prefix'].lower()} for subtask {subtask_name}. This task focuses on: {subdataset_descriptions.get(subtask_name, 'No description available.')}",
            )
            
            fields_for_this_recordset = []
            
            # You can dynamically create fields by reading CSV headers (e.g., using pandas)
            for col_name in pd.read_csv(file_path, nrows=0).columns:
                dyn_source = mlc.Source(
                   extract=mlc.Extract(column=col_name),
                   file_object=file_object.id # Pass the ID of the file_object
                )
                datatype = mlc.DataType.INTEGER if "label" in col_name.lower() or "gt" in col_name.lower() else mlc.DataType.FLOAT32
                dyn_field = mlc.Field(name=col_name.lower().replace(" ","_"), data_types=[datatype], source=dyn_source)
                fields_for_this_recordset.append(dyn_field)

            for f_obj in fields_for_this_recordset:
                 record_set.fields.append(f_obj) # Add created fields to the RecordSet

            metadata.record_sets.append(record_set) # Add RecordSet to the dataset

    # Validate the created metadata (optional but recommended)
    # The `issues` object will contain any errors or warnings found.
    issues = metadata.issues
    if len(issues.errors) > 0:
        print("\nCroissant metadata has ERRORS:")
        print(issues.report())
        # You might want to stop here if there are errors:
        # return
    elif len(issues.warnings) > 0:
        print("\nCroissant metadata has WARNINGS:")
        print(issues.report())

    # Convert the Metadata object to a JSON-LD dictionary and save it
    try:
        json_ld_output_dict = metadata.to_json()
        with open(os.path.abspath(output_json_file), "w") as f:
            json.dump(json_ld_output_dict, f, indent=2)
        print(f"\nSuccessfully created Croissant dataset definition at: {os.path.abspath(output_json_file)}")
        print("\n***********************************************************************************")
        print("IMPORTANT: Please open and review the generated JSON file, especially:")
        print("  - `name` for Fields (e.g., 'label_column', 'feature_1')")
        print("  - `dataTypes` for Fields (e.g., 'sc:Text', 'sc:Integer')")
        print("  - `extract.column` within Field sources (must match your CSV headers exactly!)")
        print("***********************************************************************************")

    except Exception as e:
        print(f"Error during Croissant metadata JSON generation or saving: {e}")
        # If validation was done, issues might provide more context
        if hasattr(metadata, 'issues') and len(issues.errors.union(issues.warnings)) > 0:
            print("Review the previously printed issues report for clues.")
        raise


# --- How to use ---
if __name__ == "__main__":
    # --- CONFIGURATION ---
    # 1. Set the path to your main folder containing all subtask folders
    #    For testing, this uses the dummy path created above.
    #    CHANGE THIS to your actual path for real data.
    path_to_your_subtasks = "/home/dominik/Documents/repos/OpenHands/evaluation/benchmarks/error_bench/tasks/explorative_data_analysis"
    # Example for real data: path_to_your_subtasks = "/path/to/your/actual/subtasks_root_folder"

    # 2. Set the desired output path for the Croissant JSON-LD file
    #    It's common to place this metadata.json file at the root of your dataset,
    #    or in a dedicated 'metadata' folder.
    #    This example places it one level above the 'subtasks' folder.
    output_croissant_json_ld_file = os.path.join(
        os.path.dirname(os.path.abspath(path_to_your_subtasks)), # Puts it in the parent of `dummy_base_path`
        "EDAx17_croissant.json"
    )
    # Alternative: place it inside the subtasks root folder:
    # output_croissant_json_ld_file = os.path.join(path_to_your_subtasks, "metadata.json")


    # Check if the path_to_your_subtasks exists before proceeding
    if not os.path.isdir(path_to_your_subtasks):
        print(f"Error: The specified base_path '{path_to_your_subtasks}' does not exist or is not a directory.")
        print("If using the dummy path, ensure the script ran the dummy data creation part.")
        print("Otherwise, please update 'path_to_your_subtasks' to your actual data location.")
    else:
        print(f"\nUsing base path for subtasks: {os.path.abspath(path_to_your_subtasks)}")
        print(f"Will generate Croissant JSON-LD at: {os.path.abspath(output_croissant_json_ld_file)}")
        
        create_croissant_dataset_with_library(path_to_your_subtasks, output_croissant_json_ld_file)


    # try to load the generated JSON file
    dataset = mlc.Dataset(jsonld=output_croissant_json_ld_file)
    available_records = dataset.metadata.record_sets
    print(f"\nLoaded dataset with {len(available_records)} record sets.")
    for record_set in available_records:
        print(f"Record set ID: {record_set.id}, Name: {record_set.name}")
        record = dataset.records(record_set.id)
        print("Record", record)