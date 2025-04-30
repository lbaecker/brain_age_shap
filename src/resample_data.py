"""Script to assess distribution of training data and undersample Biobank, where needed.

This script does the following before and after resampling:
- saves sample sizes
- plots age and gender distribution

Optional steps:
- remove small ages (before resampling)
- sigma clipping (after random resampling)
"""
from pathlib import Path

from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.utils import resample
from tableone import TableOne

from src.definitions import COLUMNS_NAME, DISEASE_DICT
from src.utils import get_sample_sizes, histogram_plot, histogram_plot_bottomlegend, violin_plot_gender_age

import warnings
from pandas.errors import PerformanceWarning

warnings.simplefilter("ignore", PerformanceWarning)
warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)

PROJECT_ROOT = Path.cwd()


def undersample_group(df_group, create_vars, label_vars, n_class, random_state):
    """Undersample a DataFrame by age using RandomUnderSampler, based on a specific class size.

    Parameters
    ----------
    df_group : DataFrame
        DataFrame to be undersampled.
    create_vars : list of str
        List of column names to use as features.
    label_vars : list of str
        List of column names that are taken into account when resampling (e.g., Age).
    n_class : int
        Subjects per age group after resampling.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    df_resampled : DataFrame
        Resampled DataFrame.
    """
    X, y = df_group[create_vars], df_group[label_vars]

    # Create dictionary of age groups to undersample based on predetermined threshold n_class
    larger = df_group.Age.value_counts() > n_class
    filtered_values_larger = df_group.Age.value_counts().loc[larger].index
    filtered_df_larger = df_group[df_group.Age.isin(filtered_values_larger)]

    sampling_strategy_dict_larger = {age: n_class for age in filtered_df_larger.Age.unique()}

    # Define undersampling function
    under = RandomUnderSampler(sampling_strategy=sampling_strategy_dict_larger, random_state=random_state)

    # Implement undersampling
    X_resampled, y_resampled = under.fit_resample(X, y)

    return pd.concat([X_resampled, y_resampled], axis=1)


def undersample_biobank(df, create_vars, label_vars=['Age'], n_class=100, random_state=243, split_by_gender=False):
    """Undersample UK Biobank-Scanner01 dataset to ensure a more even age distribution, with optional gender balancing.

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame with data.
    create_vars : list of str
        List of column names to use as features.
    label_vars : list of str, default=['Age']
        List of column names that are taken into account when resampling, e.g. Age.
    n_class : int, default=100
        Subjects per age group after resampling.
    random_state : int, default=243
        Random seed for reproducibility.
    split_by_gender : bool, default=False
        If True, ensures equal sampling from men and women.

    Returns
    -------
    df_resampled : DataFrame
        Resampled pandas DataFrame.
    """
    # Select Biobank scanner, apply undersampling, then merge to original again
    df_biobank = df[df.scanner == 'BIOBANK-SCANNER01']
    df_no_biobank = df[df.scanner != 'BIOBANK-SCANNER01']
    print('The sample size of BIOBANK-SCANNER01 before resampling is ', df_biobank.shape[0])

    # If split_by_gender is True, undersample men and women separately
    if split_by_gender:
        df_women = df_biobank[df_biobank.Gender == 'Female']
        df_men = df_biobank[df_biobank.Gender == 'Male']

        # Undersample men and women separately
        df_men_resampled = undersample_group(df_men, create_vars, label_vars, n_class // 2, random_state)
        df_women_resampled = undersample_group(df_women, create_vars, label_vars, n_class // 2, random_state)

        # Concatenate resampled men and women DataFrames
        df_biobank_resampled = pd.concat([df_men_resampled, df_women_resampled])

    else:
        # Undersample without splitting by gender
        df_biobank_resampled = undersample_group(df_biobank, create_vars, label_vars, n_class, random_state)

    # Restructure columns before merge (scanner, ID)
    df_biobank_resampled = df_biobank_resampled.assign(
        scanner='BIOBANK-SCANNER01',
        participant_id=df_biobank_resampled.id.astype(str),
        id=lambda df: df['dataset'].astype(str) + '_' + df['participant_id'].astype(str)
    )

    df_biobank_resampled.loc[:, 'groups'] = 'train_ids'

    # Ensuring defragmentation for performance
    df_biobank_resampled = df_biobank_resampled.copy()

    print('The sample size of BIOBANK-SCANNER01 after resampling is ', df_biobank_resampled.shape[0])

    # Merge resampled Biobank df into the original dataframe containing all datasets
    df_resampled = pd.concat([df_biobank_resampled, df_no_biobank], axis=0)
    df_resampled.reset_index(inplace=True, drop=True)  # Reset index

    return df_resampled


def stratified_undersample_to_common_age_range(data_subset, group_column='Diagn', age_column='Age', n_bins=5, random_state=None):
    """Stratifies and undersamples data within a common age range to match the age distribution
    across groups, balancing the sample sizes within each age bin.

    Parameters:
    - data_subset (DataFrame): The subset of data for a specific analysis category.
    - group_column (str): The column name for the grouping variable (e.g., 'Diagn' with 1 as controls and others as patients).
    - age_column (str): The column name for the age variable.
    - n_bins (int): Number of age bins for stratified undersampling.
    - random_state (int, optional): Random seed for reproducibility in undersampling.

    Returns:
    - DataFrame: A new DataFrame with stratified undersampling based on age bins and balanced sample sizes.
    """
    # Separate patients and controls
    controls = data_subset[data_subset[group_column] == 1]
    patients = data_subset[data_subset[group_column] != 1]

    # Calculate and print initial mean ages and sample sizes
    print("Before undersampling:")
    print(f"Mean age - Controls: {controls[age_column].mean():.2f}, Patients: {patients[age_column].mean():.2f}")
    print(f"Sample size - Controls: {controls.shape[0]}, Patients: {patients.shape[0]}")

    # Conduct t-test before undersampling
    t_stat_before, p_value_before = ttest_ind(controls[age_column], patients[age_column], equal_var=False)
    print(f"T-test before undersampling: t-statistic = {t_stat_before:.2f}, p-value = {p_value_before:.4f}")

    # Define the common age range
    common_min_age = max(controls[age_column].min(), patients[age_column].min())
    common_max_age = min(controls[age_column].max(), patients[age_column].max())
    print(f"Common Age Range: {common_min_age} to {common_max_age}")

    # Filter both groups to the common age range
    controls = controls[(controls[age_column] >= common_min_age) & (controls[age_column] <= common_max_age)]
    patients = patients[(patients[age_column] >= common_min_age) & (patients[age_column] <= common_max_age)]

    # Define age bins within the common age range
    age_bins = np.linspace(common_min_age, common_max_age, n_bins + 1)

    # Container for balanced data
    balanced_data = pd.DataFrame()

    # Stratified undersampling within each age bin
    for i in range(n_bins):
        bin_min, bin_max = age_bins[i], age_bins[i + 1]
        bin_controls = controls[(controls[age_column] >= bin_min) & (controls[age_column] < bin_max)]
        bin_patients = patients[(patients[age_column] >= bin_min) & (patients[age_column] < bin_max)]

        # Skip bin if either group has no samples in this age bin
        if len(bin_controls) == 0 or len(bin_patients) == 0:
            continue

        # Determine the target size for undersampling (smallest bin count)
        target_size = min(len(bin_controls), len(bin_patients))

        # Undersample within this age bin
        bin_controls_sampled = resample(bin_controls, replace=False, n_samples=target_size, random_state=random_state)
        bin_patients_sampled = resample(bin_patients, replace=False, n_samples=target_size, random_state=random_state)

        # Concatenate sampled data
        balanced_data = pd.concat([balanced_data, bin_controls_sampled, bin_patients_sampled], ignore_index=True)

    # Calculate and print mean ages and sample sizes after undersampling
    controls_balanced = balanced_data[balanced_data[group_column] == 1]
    patients_balanced = balanced_data[balanced_data[group_column] != 1]
    print("After undersampling:")
    print(
        f"Mean age - Controls: {controls_balanced[age_column].mean():.2f}, Patients: {patients_balanced[age_column].mean():.2f}")
    print(f"Sample size - Controls: {controls_balanced.shape[0]}, Patients: {patients_balanced.shape[0]}")

    # Conduct t-test after undersampling
    t_stat_after, p_value_after = ttest_ind(
        controls_balanced[age_column],
        patients_balanced[age_column],
        equal_var=False
    )
    print(f"T-test after undersampling: t-statistic = {t_stat_after:.2f}, p-value = {p_value_after:.4f}")
    print("-" * 50)

    return balanced_data


def sigma_clipping_exclude(dataframe, features=COLUMNS_NAME, n_sigmas=3, n_extreme=10):
    """Exclude subjects with too many extreme measurements in the brain regional volumes.

    Parameters
    ----------
    dataset : NDFrame of shape [n_subjects, n_features]
        DataFrame with the subject data.
    features : list, default=vars.COLUMNS_NAME
        List of the features to be considered on the effect size test. Default are the 101 FreeSurfer regions.
    n_sigmas : float, default=3
        Threshold for excluding subsets in standard deviations from the mean.
    n_extreme : int, default=10
        Number of features out of the `n_sigmas` range.

    Returns
    -------
    dataframe : NDFrame of shape [n_subjects, n_features]
        DataFrame with the cleaned dataset.
    """
    delta_n = 1
    while delta_n != 0:
        n_0 = len(dataframe)
        hc_filt = dataframe["Diagn"] == 1
        mean = dataframe[hc_filt][features].mean(axis=0)
        std = dataframe[hc_filt][features].std(axis=0)
        clip_filt = ((dataframe[features] - mean) ** 2) ** 0.5 > n_sigmas * std
        clip_filt = clip_filt.sum(axis=1) > n_extreme
        dataframe = dataframe[~clip_filt]
        delta_n = n_0 - len(dataframe)
    return dataframe


def resample_data(data_path, n_biobank=100):
    """Resample training data to obtain a more even age distribution.

    Parameters
    ----------
    data_path : Path
        Path to the dataset created in fetch_data.py.
    n_biobank : int, default=60
        Number of subjects per age group in Biobank-Scanner01 to undersample to.

    Returns
    -------
    Outputs:

    filename_path : Path
        Path where dataset_resampled.csv is saved.
    """
    # Define input and output directories
    data_dir = Path(PROJECT_ROOT / 'data')
    plot_dir = data_dir / 'resample_plots'
    plot_dir.mkdir(exist_ok=True)

    # Load dataset file containing demographic as well as FreeSurfer data
    dataset = pd.read_csv(data_path)

    # Rename gender column for easier labeling in figures
    dataset.Gender = dataset.Gender.map({0: "Female", 1: "Male"})
    dataset.loc[:, 'Diagn_labels'] = dataset['Diagn'].map(DISEASE_DICT)

    # Split into train and validation data
    # Only the training data are resampled
    dataset_train = dataset[dataset.groups == 'train_ids']
    dataset_val = dataset[dataset.groups == 'validation_ids']

    # -----------------------------------------------------------------------------------------------
    # Resampling by undersampling the largest scanner, Biobank Scanner1; the rest stays the same
    create_vars = COLUMNS_NAME + ["EstimatedTotalIntraCranialVol",
                                  "dataset", 'mriqc_prob', 'Gender', 'Diagn', 'Diagn_labels', 'id']
    data_undersampled = undersample_biobank(dataset_train,
                                            create_vars,
                                            ['Age'],
                                            n_class=n_biobank,
                                            random_state=42,
                                            split_by_gender=True)
    # Plot training data by GENDER before and after undersampling
    # Please note: This is not the final dataset, as more data cleaning steps are done after this
    plot_undersample_biobank = histogram_plot(df_original=dataset_train,
                                              df_resampled=data_undersampled,
                                              output_dir=plot_dir,
                                              filename='plot_undersample_biobank.png',
                                              title_top='Training data before data cleaning',
                                              title_bottom='Training data after undersampling UK Biobank-Scanner01',
                                              x="Age",
                                              hue="Gender",
                                              stacked=True,
                                              include_two_legends=False)

    resampled = data_undersampled

    # Save sample sizes after resampling, before sigma clipping
    total_df_before_clipping, n_train_scanner_df_before_clipping, n_val_scanner_df_before_clipping, n_val_diagnosis_df_before_clipping = get_sample_sizes(
        resampled, dataset_val, output_path=data_dir, dir_name='n_dataset_resampled')

    # Plot training data by SCANNER before and after resampling
    # Get unique scanner names and sort them alphabetically
    scanner_names = sorted(dataset_train['scanner'].unique())
    n_scanner_categories = len(scanner_names)

    # Create a color palette using seaborn
    # COLORS_SCANNER = sns.color_palette('Paired', n_colors=n_scanner_categories)
    COLORS_SCANNER = sns.color_palette('husl', n_colors=n_scanner_categories)

    # Create a dictionary mapping each scanner name to its corresponding color
    # This ensures that the scanner colors are consistent between the two plots
    scanner_color_mapping_train = dict(zip(scanner_names, COLORS_SCANNER))

    # Pass the color mapping to plot function for both datasets
    histogram_age_scanner = histogram_plot(
        df_original=dataset_train,
        df_resampled=resampled,
        output_dir=plot_dir,
        filename='histogram_age_scanner_TRAIN_undersampled.png',
        title_top='Training data before data cleaning',
        title_bottom='Training data after undersampling UK Biobank-Scanner01',
        x="Age",
        hue="scanner",
        stacked=True,
        colors=[scanner_color_mapping_train[name] for name in scanner_names],
        colors2=[scanner_color_mapping_train[name]
                 for name in scanner_names],
        include_two_legends=False
    )

    # ---------------------------------------------------------
    print('************** POOLING DIAGNOSTIC GROUPS IN VALIDATION DATA **************')
    # Pool data by clinical groups instead of datasets
    # Note: HC may be in multiple group comparisons, if the dataset contained more than one diagnosis

    # Create a new column to store which analysis comparison (diagn vs HC) a subject belongs to in the pooled groups
    dataset_val.loc[:, 'analysis_categories'] = np.empty(len(dataset_val), dtype=object)
    dataset_val.loc[:, 'analysis_categories'] = dataset_val['analysis_categories'].apply(lambda x: [])

    # Loop over each dataset, and each Diagn within it, and add the diagnostic comparison in the analysis_categories column
    for dataset_name in dataset_val['dataset'].unique():
        # Subset the DataFrame to the current dataset
        df_subset = dataset_val[dataset_val['dataset'] == dataset_name]

        # Identify unique diagnostic groups within this dataset
        diagnostic_groups = df_subset['Diagn'].unique()

        for diagnosis in diagnostic_groups:
            if diagnosis != 1:
                is_patient = (dataset_val['dataset'] == dataset_name) & (dataset_val['Diagn'] == diagnosis)
                is_control = (dataset_val['dataset'] == dataset_name) & (dataset_val['Diagn'] == 1)

                # Labeling both patients and corresponding controls as the same analysis
                dataset_val.loc[is_patient, 'analysis_categories'] = dataset_val.loc[is_patient, 'analysis_categories'].apply(
                    lambda x: x + [f"{diagnosis}_comparison"])
                dataset_val.loc[is_control, 'analysis_categories'] = dataset_val.loc[is_control, 'analysis_categories'].apply(
                    lambda x: x + [f"{diagnosis}_comparison"])

    # Verify the result by printing a sample
    # print(dataset_val[['dataset', 'Diagn', 'analysis_categories']].head())

    # Explode lists in 'group' column so that each element gets its own row
    dataset_val = dataset_val.explode('analysis_categories')
    dataset_val['analysis_categories'] = dataset_val['analysis_categories'].astype(str)

    # ---------------------------------------------------------
    # Create a new column with diagnosis labels to include in figure legends
    # dataset_val.loc[:, 'Diagn_labels'] = dataset_val['Diagn'].map(DISEASE_DICT)
    # dataset_val.Diagn_labels.isna().sum()

    # ---------------------------------------------------------
    print('************** UNDERSAMPLING VALIDATION DATA **************')
    # Undersample to reduce significant age difference between groups using age range

    # List to store the balanced data subsets
    balanced_data_list = []

    # Loop through each unique category in `analysis_categories`
    for analysis_name in dataset_val['analysis_categories'].unique():
        print(f"Category: {analysis_name}")

        # Create a subset of the data for the current category
        data_subset = dataset_val[dataset_val['analysis_categories'] == analysis_name]

        # Apply the undersampling function to this subset
        balanced_data_subset = stratified_undersample_to_common_age_range(
            data_subset, group_column='Diagn', age_column='Age', n_bins=5, random_state=42)

        # Append the balanced subset to the list
        balanced_data_list.append(balanced_data_subset)

    # Concatenate all balanced subsets into a single DataFrame
    dataset_val_balanced = pd.concat(balanced_data_list, ignore_index=True)

    # Re-group exploded items, so there are no duplicate IDs
    # Note that any future analysis of patient-HC comparisons in the validation data will need exploding again
    # Identify all columns except 'analysis_categories'
    group_columns = [col for col in dataset_val_balanced.columns if col != 'analysis_categories']

    dataset_val_balanced = dataset_val_balanced.copy()
    dataset_val_balanced_regrouped = (
        dataset_val_balanced.groupby(group_columns, as_index=False)
        .agg({'analysis_categories': lambda x: list(x.unique())})
    )

    # Display the final DataFrame with recomposed 'analysis_categories'
    # print(dataset_val_balanced.shape, dataset_val_balanced_regrouped.shape)
    # dataset_val_balanced.id.value_counts()
    # dataset_val_balanced_regrouped.id.value_counts()

    # Drop small validation scanners: drop val datasets < 5 subjects to enable harmonisation later
    scanner_counts_val = dataset_val_balanced_regrouped.scanner.value_counts()
    valid_val_scanners = scanner_counts_val[scanner_counts_val >= 5].index.tolist()
    dataset_val_balanced_regrouped_2 = dataset_val_balanced_regrouped[dataset_val_balanced_regrouped['scanner'].isin(
        valid_val_scanners)]

    # print(resampled.shape[0], dataset_val_balanced_regrouped.shape[0])
    # print(resampled.shape[0], dataset_val_balanced_regrouped_2.shape[0])

    subjects_removed = dataset_val_balanced_regrouped.shape[0] - dataset_val_balanced_regrouped_2.shape[0]
    print("Removing subjects from small scanners in validation after resampling, n=", subjects_removed)

    # ---------------------------------------------------------
    print('************** PERFORM SIGMA CLIPPING **************')
    # Perform sigma clipping on resampled training dataset
    print("Sample size of training data BEFORE sigma clipping: ", resampled.shape[0])
    resampled_clipped = sigma_clipping_exclude(resampled, features=COLUMNS_NAME, n_sigmas=3, n_extreme=10)
    print("Sample size of training data AFTER sigma clipping: ", resampled_clipped.shape[0])

    # Perform sigma clipping on healthy controls in validation/clinical datasets (incl. exploded rows)
    dataset_val_hc = dataset_val_balanced_regrouped_2[dataset_val_balanced_regrouped_2.Diagn == 1]
    dataset_val_non_hc = dataset_val_balanced_regrouped_2[dataset_val_balanced_regrouped_2.Diagn != 1]

    print("Sample size of HEALTHY CONTROLS in clinical data BEFORE sigma clipping: ",
          dataset_val_hc.shape[0])
    dataset_val_hc_clipped = sigma_clipping_exclude(dataset_val_hc, features=COLUMNS_NAME, n_sigmas=3, n_extreme=10)
    print("Sample size of HEALTHY CONTROLS in clinical data AFTER sigma clipping: ",
          dataset_val_hc_clipped.shape[0])

    # Merge clipped validation HC with validation patients
    dataset_val_clipped = pd.concat([dataset_val_hc_clipped, dataset_val_non_hc], axis=0)
    dataset_val_clipped.reset_index(inplace=True, drop=True)

    # ----------------------------------------------------------
    # Create tableone for age and sex difference in resampled balanced validation data

    # Select the relevant columns and rows for inclusion in table
    columns = ['Age', 'Gender', 'Diagn_labels']
    categorical = ['Gender', 'Diagn_labels']
    groupby = 'Diagn_labels'

    # Create the same tables but grouped by analysis categories instead of datasets
    # Define the output file path for the demographic tables
    table2_output_file = os.path.join(plot_dir, "table2_val_resampled_clipped.csv")
    age_stats_output_file = os.path.join(plot_dir, "age_stats_val_resampled_clipped.csv")

    # Clear any previous content in both files by opening in write mode
    with open(table2_output_file, 'w') as f:
        f.write('')

    with open(age_stats_output_file, 'w') as f:
        f.write('')

    dataset_for_table2 = dataset_val_clipped.explode('analysis_categories')
    print(dataset_val_clipped.shape, dataset_for_table2.shape)

    dataset_for_table2.Diagn_labels.isna().sum()
    # Loop over each unique analysis category and generate a TableOne and age stats for each
    for analysis_name in dataset_for_table2['analysis_categories'].unique():
        # Filter the dataframe for the current analysis category
        df_subset = dataset_for_table2[dataset_for_table2['analysis_categories'] == analysis_name]

        # Generate the TableOne for the current analysis category
        table2 = TableOne(df_subset, columns=columns, categorical=categorical, groupby=groupby, pval=True)

        # Append analysis category name and table2 to the combined CSV file
        with open(table2_output_file, 'a') as f:
            f.write(f"Analysis Category: {analysis_name}\n")
            table2.to_csv(f)
            f.write("\n")

        # Calculate mean age and age range per comparison for age_stats
        age_stats = df_subset.groupby('Diagn_labels')['Age'].agg(['mean', 'min', 'max']).round(1)

        # Append analysis category name and age_stats to the age stats CSV file
        with open(age_stats_output_file, 'a') as f:
            f.write(f"Analysis Category: {analysis_name}\n")
            age_stats.to_csv(f)
            f.write("\n")

    # ---------------------------------------------------------
    print('************** PLOT ORIGINAL VS FINAL DATASETS **************')
    # Plot training data by scanner
    histogram_age_scanner_train_final = histogram_plot(
        df_original=dataset_train,
        df_resampled=resampled_clipped,
        output_dir=plot_dir,
        filename='histogram_age_scanner_TRAIN_FINAL.png',
        title_top='Training data before data cleaning',
        title_bottom='Training data after undersampling and sigma clipping',
        x="Age",
        hue="scanner",
        stacked=True,
        colors=scanner_color_mapping_train,
        colors2=scanner_color_mapping_train,
        include_two_legends=False
    )

    # Plot validation data by scanner
    scanner_names_val = sorted(dataset_val['scanner'].unique())
    n_scanner_categories_val = len(scanner_names_val)
    COLORS_SCANNER_val = sns.color_palette('husl', n_colors=n_scanner_categories_val)
    scanner_color_mapping_val = dict(zip(scanner_names_val, COLORS_SCANNER_val))

    # Pass the colors in the correct order (align with sorted scanner names)
    histogram_age_scanner_val_final = histogram_plot(
        df_original=dataset_val,
        df_resampled=dataset_val_clipped,
        output_dir=plot_dir,
        filename='histogram_age_scanner_VALIDATION_FINAL.png',
        title_top='Clinical data before data cleaning',
        title_bottom='Clinical data after undersampling and sigma clipping',
        x="Age",
        hue="scanner",
        stacked=True,
        colors=[scanner_color_mapping_val[name] for name in scanner_names_val],
        colors2=[scanner_color_mapping_val[name] for name in scanner_names_val],
        include_two_legends=False
    )

    # Plot validation data by diagnosis
    diagnosis_names_val = sorted(dataset_val['Diagn_labels'].unique())
    n_diagnosis_categories_val = len(diagnosis_names_val)
    COLORS_SCANNER_val_diagnosis = sns.color_palette('tab20', n_colors=n_diagnosis_categories_val)
    # COLORS_SCANNER_val_diagnosis = sns.color_palette('husl', n_colors=n_diagnosis_categories_val)
    diagnosis_color_mapping_val = dict(zip(diagnosis_names_val, COLORS_SCANNER_val_diagnosis))

    # Pass the colors in the correct order (align with sorted scanner names)
    histogram_age_diagnosis_val_final = histogram_plot(
        df_original=dataset_val,
        df_resampled=dataset_val_clipped,
        output_dir=plot_dir,
        filename='histogram_age_diagnosis_VALIDATION_FINAL.png',
        title_top='Clinical data before data cleaning',
        title_bottom='Clinical data after undersampling and sigma clipping',
        x="Age",
        hue="Diagn_labels",
        stacked=True,
        colors=[diagnosis_color_mapping_val[name] for name in diagnosis_names_val],
        colors2=[diagnosis_color_mapping_val[name] for name in diagnosis_names_val],
        include_two_legends=False
    )

    # Plot final training vs validation data on same scale split by GENDER
    gender_names = sorted(dataset['Gender'].unique())
    n_gender_categories = len(gender_names)
    COLORS_SCANNER_gender = sns.color_palette('tab20', n_colors=n_gender_categories)
    # COLORS_SCANNER_gender = sns.color_palette('husl', n_colors=n_gender_categories)
    gender_color_mapping = dict(zip(gender_names, COLORS_SCANNER_gender))

    # Pass the colors in the correct order (align with sorted scanner names)
    histogram_age_gender_total_final = histogram_plot(
        df_original=resampled_clipped,
        df_resampled=dataset_val_clipped,
        output_dir=plot_dir,
        filename='histogram_age_gender_total_FINAL.png',
        title_top='Training data',
        title_bottom='Clinical data',
        x="Age",
        hue="Gender",
        stacked=True,
        colors=[gender_color_mapping[name] for name in gender_names],
        colors2=[gender_color_mapping[name] for name in gender_names],
        include_two_legends=False
    )

    # Plot final training vs validation data on same scale split by DIAGNOSIS
    diagnosis_names = sorted(dataset['Diagn_labels'].unique())
    n_diagnosis_categories = len(diagnosis_names)
    COLORS_SCANNER_diagnosis = sns.color_palette('tab20', n_colors=n_diagnosis_categories)
    # COLORS_SCANNER_diagnosis = sns.color_palette('husl', n_colors=n_diagnosis_categories)
    diagnosis_color_mapping = dict(zip(diagnosis_names, COLORS_SCANNER_diagnosis))

    # Pass the colors in the correct order (align with sorted scanner names)
    histogram_age_diagnosis_total_final = histogram_plot_bottomlegend(
        df_original=resampled_clipped,
        df_resampled=dataset_val_clipped,
        output_dir=plot_dir,
        filename='histogram_age_diagnosis_total_FINAL.png',
        title_top='Training data',
        title_bottom='Clinical data',
        x="Age",
        hue="Diagn_labels",
        stacked=True,
        colors=[diagnosis_color_mapping[name] for name in diagnosis_names],
        colors2=[diagnosis_color_mapping[name] for name in diagnosis_names],
        include_two_legends=False
    )

    # ---------------------------------------------------------
    print('************** SAVING DATASET **************')
    # Save sample sizes after resampling and sigma clipping

    total_df, n_train_scanner_df, n_val_scanner_df, n_val_diagnosis_df = get_sample_sizes(
        resampled_clipped, dataset_val_clipped, output_path=data_dir, dir_name='n_dataset_resampled_clipped')

    # Merge resampled_clipped with dataset_val_clipped and save that as a final dataset to input to harmonisation
    resampled_clipped.drop(columns=['participant_id'], inplace=True)
    dataset_final = pd.concat([resampled_clipped, dataset_val_clipped], ignore_index=True)

    # Save resampled dataset
    filename = 'dataset_resampled.csv'
    file_path = data_dir / filename
    dataset_final.to_csv(file_path)

    # Sample size check to see if it matches import in run_combat
    dataset_train = dataset_final[dataset_final.groups == 'train_ids']
    dataset_val = dataset_final[dataset_final.groups == 'validation_ids']

    print("Sample size of training data to be saved: ", dataset_train.shape[0])
    print("Sample size of validation data to be saved: ", dataset_val.shape[0])
    print("Total sample size (training + validation): ", dataset_train.shape[0] + dataset_val.shape[0])
    print("Total sample size (dataset_final): ", dataset_final.shape[0])

    return file_path


if __name__ == "__main__":
    resample_data()
