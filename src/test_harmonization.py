"""Script to visualise the results of the ComBat scanner harmonisation.

This is done using the Kolmogorov-Smirnov test, as done for Neuroharmony previously (Garcia-Dias et al. 2021).
"""
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from scipy.special import comb
from matplotlib.colors import LogNorm
from pathlib import Path

from src.definitions import COLUMNS_NAME

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = Path.cwd()


def check_vars(df, features):
    """Check if all variables in a list are present in a dataframe."""
    features = pd.Series(features)
    is_feature_present = features.isin(df.columns)
    missing_features_str = "Missing features: %s" % ', '.join(features[~is_feature_present])
    assert is_feature_present.all(), ValueError(missing_features_str)


def compare_dfs(ks_original, ks_harmonized):
    """Compare two dictionaries of dataframes to measure the success of harmonization.

    Function from Neuroharmony repo.
    """
    vars = ks_original.keys()
    var_improved = pd.Series(index=vars, dtype="bool")
    for var in vars:
        improved = ks_original[var] < ks_harmonized[var]
        var_improved.loc[var] = improved.sum().sum()
    return var_improved


def ks_test_grid(df, features, sampling_variable='scanner'):
    """Calculate the Kolmogorov-Smirnov score for all pairs of scanners.

    Function copied from Neuroharmony github repo:
    https://github.com/garciadias/Neuroharmony/blob/master/neuroharmony/models/metrics.py

    Parameters
    ----------
    df: NDFrame of shape [n_subjects, n_features]
        DataFrame with the subjects data.

    features: list
        List of the features to be considered on the Kolmogorov-Smirnov test.

    sampling_variable: str, default='scanner'
        Variable for which you want to group subjects.

    Returns
    -------
    KS_by_variable: dict of NDFrames
        Kolmogorov-Smirnov p-values to all pairs of instances in the sampling_variable column.
        The keys in the dictionary are the variables in 'features'. The values of each entry are square NDFrames of
        shape [n_vars, n_vars].

    Raises
    ------
    ValueError:
        If the list of variables contain any variable that is not present in df.

    Examples
    --------
    >>> ixi = DataSet('data/raw/IXI').data
    >>> features = ['Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', ]
    >>> KS = ks_test_grid(df, features, 'scanner')
    >>> KS[features[0]]
    +--------------------------+----------------------+------------------------+--------------------+
    |                          | SCANNER01-SCANNER01  | SCANNER02-SCANNER01    | SCANNER03-SCANNER01|
    +==========================++=====================+========================+====================+
    |SCANNER01-SCANNER01       | NaN                  | NaN                    | NaN                |
    +--------------------------+----------------------+------------------------+--------------------+
    |SCANNER02-SCANNER01       | 0.000759473          | NaN                    | NaN                |
    +--------------------------+----------------------+------------------------+--------------------+
    |SCANNER03-SCANNER01       | 0.0539998            | 0.625887               | NaN                |
    +--------------------------+----------------------+------------------------+--------------------+
    """
    check_vars(df, features)
    groups = df.groupby(sampling_variable)
    scanners_list = df[sampling_variable].unique()
    scanners_list = scanners_list[np.argsort(df[sampling_variable].str.lower().unique())]
    KS_by_variable = {}
    for var in features:
        KS = pd.DataFrame([], index=scanners_list, columns=scanners_list)
        for scanner_batch in np.array_split(np.array(list(combinations(scanners_list, 2))), 80):
            for scanner_a, scanner_b in scanner_batch:
                group_a = groups.get_group(scanner_a)[var]
                group_b = groups.get_group(scanner_b)[var]
                KS.loc[scanner_b][scanner_a] = ks_2samp(group_a, group_b).pvalue
        KS_by_variable[var] = KS
    return KS_by_variable


def test_harmonization(data_path):
    # Load data before and after hamonisation
    # data_dir = data_path
    data_dir = Path(Path.cwd() / 'data')
    output_dir = data_dir / 'harmonisation_plots'
    output_dir.mkdir(exist_ok=True)

    filename_raw = 'dataset_resampled.csv'
    filename_harmonised = 'dataset_harmonised.csv'
    dataset_raw = pd.read_csv(data_dir / filename_raw, index_col='id')
    dataset_harmonised = pd.read_csv(data_dir / filename_harmonised, index_col='id')

    # Select HC only
    # Note: While both HC and patient data are harmonised, the scanner comparison like this only makes sense for HC
    dataset_raw_hc = dataset_raw[dataset_raw.Diagn == 1].copy()
    dataset_harmonised_hc = dataset_harmonised[dataset_harmonised.Diagn == 1].copy()

    print(dataset_raw.shape[0], dataset_harmonised.shape[0])
    print(dataset_raw_hc.shape[0], dataset_harmonised_hc.shape[0])

    # Separate into train and validation
    for i_subjects in ['train_ids', 'validation_ids']:
        # Subset the data to have before and after harmonisation
        dataset_raw_hc_subset = dataset_raw_hc[dataset_raw_hc.groups == i_subjects].copy()
        dataset_harmonised_hc_subset = dataset_harmonised_hc[dataset_harmonised_hc.groups == i_subjects].copy()
        print(dataset_raw_hc_subset.shape[0], dataset_harmonised_hc_subset.shape[0])

        # Calculate Kolmogorov-Smirnov
        KS_ORIGINAL = ks_test_grid(dataset_raw_hc_subset, COLUMNS_NAME, sampling_variable='scanner')
        KS_HARMONIZED = ks_test_grid(dataset_harmonised_hc_subset, COLUMNS_NAME, sampling_variable='scanner')

        n_datasets = KS_ORIGINAL[COLUMNS_NAME[0]].shape[0]
        KS_ARRAY_HARMONIZED = np.zeros((n_datasets, n_datasets, len(COLUMNS_NAME)))
        KS_ARRAY_ORIGINAL = np.zeros((n_datasets, n_datasets, len(COLUMNS_NAME)))

        # Populate KS_ARRAY for both harmonized and original datasets
        for i_var, var in enumerate(COLUMNS_NAME):
            KS_ARRAY_HARMONIZED[:, :, i_var] = KS_HARMONIZED[var]
            KS_ARRAY_ORIGINAL[:, :, i_var] = KS_ORIGINAL[var]

        # Get the median KS test p-values across all ROIs (columns of the array) for each pair of datasets
        # Note: The Neuroharmony publication used minimum instead of median, so it was more conservative
        datasets = KS_HARMONIZED[COLUMNS_NAME[0]].index.tolist()
        MEDIAN_KS_HARMONIZED = pd.DataFrame(np.median(KS_ARRAY_HARMONIZED, axis=2),
                                            index=datasets, columns=datasets).fillna(0)
        MEDIAN_KS_ORIGINAL = pd.DataFrame(np.median(KS_ARRAY_ORIGINAL, axis=2),
                                          index=datasets, columns=datasets).fillna(0)

        # Combine the two matrices using the original method
        MEDIAN_COMBINED = (MEDIAN_KS_ORIGINAL + MEDIAN_KS_HARMONIZED.T)

        # Optional: Replacing scanner names for better display
        MEDIAN_COMBINED.index = MEDIAN_COMBINED.index.str.replace('SCANNER', 'S')
        MEDIAN_COMBINED.columns = MEDIAN_COMBINED.columns.str.replace('SCANNER', 'S')

        MEDIAN_COMBINED.rename(index={'HumanConnectomeProject-Aging-Scanner01': 'HCP-A-S01'},
                               columns={'HumanConnectomeProject-Aging-Scanner01': 'HCP-A-S01'}, inplace=True)

        # TEMP TEST STATEMENT TO SEE IF empty heatmap spots contain values
        # MEDIAN_COMBINED[MEDIAN_COMBINED == 0] = 1e-10

        # Plot the heatmap with log-scaled colors
        vmin, vmax = 1e-4, 1e0
        cbar_ticks = [10**i for i in np.arange(np.log10(vmin), np.log10(vmax) + 1)]
        fig = plt.figure(figsize=(2 * 5.2283465, 1.2 * 5.2283465))
        ax = fig.add_subplot(111)
        ax = sns.heatmap(MEDIAN_COMBINED,
                         cmap='BrBG', norm=LogNorm(vmin=vmin, vmax=vmax),
                         cbar_kws=dict(ticks=cbar_ticks, pad=0.005), vmin=vmin, vmax=vmax, ax=ax)

        # Set plot title and appearance
        plt.title('Kolmogorov-Smirnov test (p-value)', fontsize=12)
        plt.tick_params(labelsize=9)  # Controls the x-axis label size

        # Adjust y-axis tick labels independently
        plt.gca().tick_params(axis='y', pad=8)  # Increase the padding for y-axis labels
        plt.minorticks_off()
        plt.subplots_adjust(left=0.175, bottom=0.33, top=0.95, right=1.075)

        # Adjust y-axis tick labels without rotation
        plt.gca().set_yticks(np.arange(0.5, len(MEDIAN_COMBINED)))
        plt.gca().set_yticklabels(MEDIAN_COMBINED.index, fontsize=8)  # Controls y-axis label size

        # Save the figure
        output_filename = i_subjects + '_KS_fulldataset_hc_only_MEDIAN.png'
        plt.savefig(output_dir / output_filename, dpi=3 * 96)
        plt.show()
