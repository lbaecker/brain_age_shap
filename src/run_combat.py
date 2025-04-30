"""Run Combat for scanner harmonisation.

This is based on the publically available Combat tool [1]
using the neurocombat-sklearn implementation written by Walter [2].

References:
[1] Fortin et al.
[2] https://github.com/Warvito/neurocombat_sklearn ; https://pypi.org/project/neurocombat-sklearn/0.1.2/
"""
import matplotlib.pyplot as plt
from neurocombat_sklearn import CombatModel
import numpy as np
import pandas as pd
from pathlib import Path
# from sklearn.model_selection import train_test_split

from src.definitions import COLUMNS_NAME

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = Path.cwd()


def plot_freesurfer_regions(df_original, df_reworked, x=COLUMNS_NAME, process_title='harmonisation'):
    """Bar plot comparing mean regional volumes in two DataFrames,
    e.g. before and after harmonisation or before and after resampling.

    Parameters
    ----------
    df_original : DataFrame
        Original pandas DataFrame.
    df_reworked : DataFrame
        Reworked pandas DataFrame.
    x : str, default=COLUMNS_NAME
        Name of variables on x axis of bar plot. Must be columns in both DataFrames.
        Default are the 101 FreeSurfer regional volumes (excl. EstimatedTotalIntraCranialVol).
    process_title : str, default='harmonisation'
        Name of process that is being visualised to be included in figure title.
    hue : str, default='Dataset'
        Name of variable to split the data by (if any). Must be a column in both DataFrames.

    Returns
    -------
    fig : plt
        Bar plot.
    """
    # Get mean regional volumes
    df_original_means = df_original[x].mean()
    df_reworked_means = df_reworked[x].mean()

    # Create bar plot
    number_columns = len(x)
    ind = np.arange(number_columns)
    width = 0.40
    fig = plt.figure(figsize=(28, 10))
    plt.bar(ind, df_original_means, width, label='Original')
    plt.bar(ind + width, df_reworked_means, width, label='Reworked')

    plt.ylabel('Mean volume')
    plt.title(f'Mean FreeSurfer volumes before and after {process_title}')
    # plt.rcParams['figure.figsize'] = (20, 10)
    plt.xticks(ind + width / 2, x, rotation=45, ha='right')
    plt.legend(loc='best')
    # plt.savefig(temp_dir / 'fs_volume_comparison.png')

    return fig


def plot_region_by_scanner(dataset, region_name, group='dataset', output_dir=PROJECT_ROOT, filename='region_by_scanner.png'):
    """Plot a region by scanner.

    Can be used to plot before and after harmonisation.

    Parameters
    ----------
    dataset : DataFrame
        Original pandas DataFrame containing the data regional and demographic data.
    region_name : str
        Name of region to be plotted, e.g. 'Left-Inf-Lat-Vent'.
    group : str, default='dataset'
        Column name of grouping variable. Must be a column in dataset.
    x : str, default=COLUMNS_NAME
    """
    grouped_means = dataset.groupby(group)[region_name].mean()
    grouped_stds = dataset.groupby(group)[region_name].std()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(grouped_means.index, grouped_means, yerr=grouped_stds, fmt='o', color='blue', label='Mean with Std Dev')
    ax.set_ylabel('Regional volume')
    ax.set_title(f'Mean volume of {region_name} with Error Bars (Standard Deviation) Grouped by {group}')
    ax.legend()

    # Show the plot
    plt.show()

    return fig


def create_combat_model(dataset):
    """Function to develop and apply a ComBat model."""
    dataset.loc[:, 'scanner_int'] = pd.factorize(dataset.loc[:, 'scanner'])[0]
    # dataset.scanner_int.value_counts()

    # Initiate model
    model = CombatModel()

    data_harmonised_nosplit = model.fit_transform(dataset[['EstimatedTotalIntraCranialVol'] + COLUMNS_NAME],
                                                  dataset[['scanner_int']],
                                                  dataset[['Diagn']],  # discrete covariate
                                                  dataset[['Age']],   # continuous covariate
                                                  )

    return data_harmonised_nosplit


def run_combat(data_path):
    """Perform scanner harmonisation using ComBat.

    Parameters
    ----------
    data_path : Path
        Path to the dataset created in resample_data.py.

    Returns
    -------
    Outputs:

    filename_path : Path
        Path where dataset_harmonised.csv is saved.
    """
    data_dir = Path(Path.cwd() / 'data')
    dataset = pd.read_csv(data_path, index_col=0)

    # Split into train and validation sets
    dataset_train = dataset[dataset.groups == 'train_ids']
    dataset_val = dataset[dataset.groups == 'validation_ids']

    # Harmonise using ComBat; done separately for training and validation data
    data_harmonised_train = create_combat_model(dataset_train)
    data_harmonised_val = create_combat_model(dataset_val)

    # Replace raw data with harmonised data (in new DF)
    dataset_harmonised_train = dataset_train.copy()
    dataset_harmonised_train[['EstimatedTotalIntraCranialVol'] + COLUMNS_NAME] = data_harmonised_train

    dataset_harmonised_val = dataset_val.copy()
    dataset_harmonised_val[['EstimatedTotalIntraCranialVol'] + COLUMNS_NAME] = data_harmonised_val

    # Combine harmonised train and validation data into one dataset
    dataset_harmonised = pd.concat([dataset_harmonised_train, dataset_harmonised_val], ignore_index=True)

    # ---------------------------------------------------------
    print('************** SAVING HARMONISED DATASET **************')

    # Save data
    filename = 'dataset_harmonised.csv'
    file_path = data_dir / filename
    dataset_harmonised.to_csv(file_path, index=False)

    print("Sample size of training data to be saved: ", dataset_harmonised_train.shape[0])
    print("Sample size of validation data to be saved: ", dataset_harmonised_val.shape[0])
    print("Total sample size (training + validation): ", dataset_harmonised.shape[0])

    return file_path


if __name__ == "__main__":
    run_combat()
