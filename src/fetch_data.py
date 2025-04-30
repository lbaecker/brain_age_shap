"""Get, organise and clean datasets.

Steps are:
- Define path where data are saved
- Define scanners to be included
- Identify files based on scanners
- Combine files into one dataset
- Remove subjects with missing values
- Separate training and clinical (test) data; training data only include healthy subjects
- Remove subjects outside of the pre-defined age range
- Remove bad-quality subjects based on MRIQC [1] rating
- Remove small scanners from training data 
(Note: functionality is still there for test data but this is not used, as those data are removed at a later stage)
- Save file

This script is based on the Neurofind script of the same name (validation_study_2020).

References:
    [1] https://mriqc.readthedocs.io/en/stable/
"""
import os
import glob
import pandas as pd
from pathlib import Path
import warnings

from src.definitions import COLUMNS_NAME
from src.utils import get_sample_sizes

PROJECT_ROOT = Path.cwd()


def get_scanner_names(dataset_list, data_dir):
    """Get scanner names."""
    bids_dir = data_dir / 'BIDS_data'
    fs_dir = data_dir / 'FreeSurfer_preprocessed'
    mriqc_dir = data_dir / 'MRIQC'

    scanner_set = set()
    scanner_bids_path_set = set()
    scanner_fs_path_set = set()
    scanner_mriqc_path_set = set()

    # Loop over datasets, get scanners from BIDS directory and append to sets
    for dataset in dataset_list:
        bids_path = bids_dir / dataset
        fs_path = fs_dir / dataset
        mriqc_path = mriqc_dir / dataset

        for subdir in os.listdir(bids_path):
            subdir_bids_path = os.path.join(bids_path, subdir)
            subdir_fs_path = os.path.join(fs_path, subdir)
            subdir_mriqc_path = os.path.join(mriqc_path, subdir)

            if os.path.isdir(subdir_bids_path):
                scanner_set.add(subdir)
                scanner_bids_path_set.add(subdir_bids_path)

            if os.path.isdir(subdir_fs_path):
                scanner_fs_path_set.add(subdir_fs_path)

            if os.path.isdir(subdir_mriqc_path):
                scanner_mriqc_path_set.add(subdir_mriqc_path)

    if len(scanner_bids_path_set) != len(scanner_fs_path_set):
        print('Did not find the same number of scanners in BIDS and FreeSurfer paths')
    elif len(scanner_bids_path_set) != len(scanner_mriqc_path_set):
        print('Did not find the same number of scanners in BIDS and MRIQC paths')
    else:
        print('Number of scanners found:', len(scanner_bids_path_set))

    return (
        list(scanner_set),
        list(scanner_bids_path_set),
        list(scanner_fs_path_set),
        list(scanner_mriqc_path_set)
    )


def fetch_data(raw_path, scanner_list, bids_path_list, fs_path_list, mriqc_path_list):
    """Combine all datasets into one file per data type incl. all data types.

    This function merges all files for demographics (participant.tsv), FreeSurfer output (freesurferData.csv)
    and MRIQC output (group_T1w.tsv) files.
    """
    # Initialise DF
    column_names = ['id', 'Age', 'Gender', 'Diagn', 'scanner',
                    'mriqc_prob', 'EstimatedTotalIntraCranialVol'] + COLUMNS_NAME
    data = pd.DataFrame(columns=column_names).set_index('id')

    # Loop over scanners
    for scanner in scanner_list:
        # get the scanner path for bids
        matching_bids_path = [path_name for path_name in bids_path_list if scanner in path_name]
        try:
            scanner_participants = pd.read_csv(
                Path(matching_bids_path[0] + '/participants.tsv'), sep='\t', index_col='image_id')
            if scanner == 'HCP-PSYCHOSIS-SCANNER01':
                scanner_participants['scanner'] = 'HCP-PSYCHOSIS-SCANNER01'
            elif 'Dataset' in scanner_participants.columns and 'scanner' not in scanner_participants.columns:
                scanner_participants.rename(columns={'Dataset': 'scanner'}, inplace=True)
            if scanner == 'IOPPN-SCANNER01':
                scanner_participants = scanner_participants.reset_index()
                scanner_participants['image_id'] = scanner_participants.apply(
                    lambda row: f"{row['participant_id']}_{row['session_id']}_{row['acq_id']}_{row['run_id']}_T1w", axis=1)
                scanner_participants = scanner_participants.set_index('image_id')
        except FileNotFoundError:
            warning_message = f"No participants file found for scanner: {scanner}"
            warnings.warn(warning_message)

        # Get the scanner path for FS
        matching_fs_path = [path_name for path_name in fs_path_list if scanner in path_name]
        try:
            scanner_fs = pd.read_csv(Path(matching_fs_path[0] + '/freesurferData.csv'), index_col='image_id')
        except FileNotFoundError:
            warning_message = f"No FreeSurfer file found for scanner: {scanner}"
            warnings.warn(warning_message)

        # Get MRIQC data
        matching_mriqc_path = [path_name for path_name in mriqc_path_list if scanner in path_name]
        matching_mriqc_path_str = str(matching_mriqc_path[0])
        files = glob.glob(matching_mriqc_path_str + '/*unseen_pred.csv')
        if files:
            scanner_mriqc = pd.read_csv(files[0], index_col='image_id')
        else:
            warning_message = f"No MRIQC file matching the pattern *unseen_pred.csv found for scanner: {scanner}"
            warnings.warn(warning_message)

        # Rename the column 'prob_y' to 'mriqc_prob', drop 'pred_y'
        scanner_mriqc = scanner_mriqc.rename(columns={'prob_y': 'mriqc_prob'})
        if 'pred_y' in scanner_mriqc.columns:
            scanner_mriqc = scanner_mriqc.drop(columns=['pred_y'])

        # Merge participants, FS data
        scanner_data = pd.merge(scanner_participants, scanner_fs, on='image_id')
        # print(f"Length of scanner_data before MRIQC merge for {scanner}: {len(scanner_data)}")
        scanner_data = pd.merge(scanner_data, scanner_mriqc, on='image_id')
        # print(f"Length of scanner_data after MRIQC merge for {scanner}: {len(scanner_data)}")

        scanner_data = pd.merge(scanner_participants, scanner_fs, on='image_id')
        scanner_data = pd.merge(scanner_data, scanner_mriqc, on='image_id')

        scanner_data = scanner_data.reset_index()
        if len(scanner_data) == 0:
            warning_message = f"No data in final data file found for scanner: {scanner}"
            warnings.warn(warning_message)

        # Define new ID that contains dataset name to avoid multiple subjects with the same ID across datasets
        scanner_data['id'] = scanner_data['scanner'] + '-' + scanner_data['participant_id']
        # scanner_data = scanner_data.set_index('id')

        # If multiple subjects have the same ID, sort by image_id or Age and include only the first one
        # Identify duplicate ids
        duplicate_ids = scanner_data[scanner_data.duplicated(subset='id', keep=False)]['id']

        # Separate the rows with duplicate ids
        duplicates_data = scanner_data[scanner_data['id'].isin(duplicate_ids)]

        # Sort duplicates by 'id' and 'Age', and keep only the first occurrence (lowest Age)
        duplicates_data = duplicates_data.sort_values(by=['id', 'Age'])
        duplicates_data = duplicates_data.drop_duplicates(subset='id', keep='first')

        # Get the rows without duplicate ids
        non_duplicates_data = scanner_data[~scanner_data['id'].isin(duplicate_ids)]

        # Concatenate the non-duplicates and filtered duplicates data back together
        scanner_data = pd.concat([non_duplicates_data, duplicates_data])

        # Concatenate to data dataframe that will contain all datasets
        data = pd.concat([data, scanner_data[column_names]])

    data = data.set_index('id')

    return data


def create_data_file(
    data_dir,
    dataset_list,
    validation_list,
    mriqc_threshold=0.7,
    min_age=7,
    max_age=85,
    min_n_per_scanner_train=50,
    min_n_per_scanner_val=1,
):
    """Create data file containing all training and validation data.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing all data.
    dataset_list : list of str
        List of dataset names that are included for training and validation.
    validation_list : list of str
        List of dataset names that are included for validation only. All must be included in dataset_list.
    mriqc_threshold : float, default=0.7
        Threshold for data quality using MRIQC tool (assessed in separate analysis). Data below this threshold are included.
    min_age : int, default=7
        Minimum age of a subject to be included (for training and validation).
    max_age : int, default=85
        Maximum age of a subject to be included (for training and validation).
    min_size : int, default=10
        Minimum number of subjects in a single scanner.
    min_n_per_scanner : int, default=50
        Minimum number of subjects per scanner in the training data.

    Returns
    -------
    Outputs:

    filename_path : Path
        Path where data_without_nan.csv is saved.
    """
    data_dir = Path(data_dir)
    dataset_list = dataset_list
    validation_list = validation_list

    output_dir = Path(Path.cwd() / 'data')
    output_dir.mkdir(exist_ok=True)

    print('************** RETRIEVING SCANNER INFORMATION **************')
    # Get scanner list, BIDS path list, and scanner FS list based on dataset list
    scanner_list, scanner_bids_path_list, scanner_fs_path_list, scanner_mriqc_path_list = get_scanner_names(
        dataset_list=dataset_list, data_dir=data_dir)

    # Create one dataset
    data = fetch_data(data_dir, scanner_list, scanner_bids_path_list, scanner_fs_path_list, scanner_mriqc_path_list)

    # Save the original sample sizes before doing any data cleaning
    original_data = data.groupby('scanner').Diagn.value_counts(dropna=False)
    df_original_data = original_data.reset_index()
    df_original_data.columns = ['scanner', 'Diagnosis', 'Count']

    df_original_data_path = output_dir / 'n_scanner_before_any_removal.csv'
    df_original_data.to_csv(df_original_data_path, index=False)

    # data.index.value_counts()
    # ---------------------------------------------------------
    print('************** DATA CLEANING AND FORMATTING **************')
    # Drop missing values in any column
    print("Sample size BEFORE dropping subjects with missing information: ", data.shape[0])
    data = data.dropna()
    print("Sample size AFTER dropping subjects with missing information: ", data.shape[0])

    # Reformat Age and Diagn columns
    # Round down age to nearest integer
    data['Age'] = data['Age'].astype(int)
    data['Diagn'] = data['Diagn'].astype(int)

    # Create new column for dataset
    # This will be used for plotting by dataset rather than scanner later
    data.loc[:, 'dataset'] = data.loc[:, 'scanner'].apply(
        lambda x: x.split('-')[0] if 'SCANNER' not in x else x.rsplit('-SCANNER', 1)[0]
    )

    # Create new column separating train_ids and validation_ids
    data.loc[:, 'groups'] = data['dataset'].apply(lambda x: 'validation_ids' if x in validation_list else 'train_ids')

    # Drop any train_ids with diagnosis != 1
    print("Sample size BEFORE dropping subjects from training data that are not HC: ", data.shape[0])
    data = data[((data['groups'] == 'train_ids') & (data['Diagn'] == 1)) | (data['groups'] == 'validation_ids')]
    print("Sample size AFTER dropping subjects from training data that are not HC: ", data.shape[0])

    # data.groupby('dataset').Diagn.value_counts()
    # ---------------------------------------------------------
    print('************** DEFINING THE AGE RANGE **************')

    # Remove subjects outside of the pre-defined age range
    print("Sample size BEFORE restricting the age range: ", data.shape[0])
    min_age = min_age
    max_age = max_age
    age_keep = data['Age'].between(min_age, max_age)
    data = data[age_keep]
    print("Sample size AFTER restricting the age range: ", data.shape[0])

    # ---------------------------------------------------------
    print('************** PERFORMING QUALITY CONTROL USING MRIQC **************')

    # Remove subjects with bad quality scans based on MRIQC rating
    print("Sample size BEFORE quality checking: ", data.shape[0])
    mriqc_threshold = mriqc_threshold
    # mriqc_threshold = 0.7
    qc_keep = (data["mriqc_prob"] < mriqc_threshold)
    data = data[qc_keep]
    print("Sample size AFTER quality checking: ", data.shape[0])

    # ---------------------------------------------------------
    print('************** REMOVING SMALL SCANNERS **************')
    # Drop any scanner with different minimum thresholds for train and validation
    min_n_per_scanner_train = min_n_per_scanner_train
    min_n_per_scanner_val = min_n_per_scanner_val

    # Create a DataFrame with counts of unique scanners
    scanner_counts_train = data[data['groups'] == 'train_ids']['scanner'].value_counts()
    scanner_counts_val = data[data['groups'] == 'validation_ids']['scanner'].value_counts()

    print("Number of train/validation scanners BEFORE removing small scanners: ",
          len(scanner_counts_train), len(scanner_counts_val))
    print("Sample size of train/validation data BEFORE removing small scanners: ",
          data[data['groups'] == 'train_ids'].shape[0], data[data['groups'] == 'validation_ids'].shape[0])

    # Get the scanners with more than min_n_per_scanner subjects
    valid_train_scanners = scanner_counts_train[scanner_counts_train >= min_n_per_scanner_train].index.tolist()
    valid_val_scanners = scanner_counts_val[scanner_counts_val >= min_n_per_scanner_val].index.tolist()

    valid_scanners = valid_train_scanners + valid_val_scanners

    data = data[data['scanner'].isin(valid_scanners)]

    # data.groupby('groups').scanner.value_counts().sort_index()
    print("Number of train/validation scanners AFTER removing small scanners: ",
          len(valid_train_scanners), len(valid_val_scanners))
    print("Sample size of train/validation data AFTER removing small scanners: ",
          data[data['groups'] == 'train_ids'].shape[0], data[data['groups'] == 'validation_ids'].shape[0])

    # ---------------------------------------------------------
    print('************** SAVING DATASET **************')

    # Save sample sizes
    dataset_train = data[data['groups'] == 'train_ids']
    dataset_validation = data[data['groups'] == 'validation_ids']
    total_df, n_train_scanner_df, n_val_scanner_df, n_val_diagnosis_df = get_sample_sizes(
        dataset_train, dataset_validation, output_path=data_dir, dir_name='n_data_without_nan')

    # Save cleaned dataset (missing values removed, irrelevant data, low-quality data)
    filename = 'data_without_nan.csv'
    filename_path = output_dir / filename
    data.to_csv(filename_path)

    return filename_path


if __name__ == '__main__':
    create_data_file()
