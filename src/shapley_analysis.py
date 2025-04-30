"""Script to analyse the Shapley values by diagnostic group.

The Shapley values are calculated in apply_model.py.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import ast
from collections import defaultdict
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection
import seaborn as sns
import matplotlib.pyplot as plt
import os

from src.definitions import COLUMNS_NAME, FS2REGION

PROJECT_ROOT = Path.cwd()


def run_shap_analysis(data_path, model_name='LinearSVR'):
    """Analyse Shapley values computed in apply_model.py.

    Parameters
    ----------
    data_path : Path
        Path to the dataset created in run_combat.py.
    model_name : str, default='LinearSVR'
        Model type for brain age prediction. Could be 'LinearSVR', 'SVR_rbf' or 'XGBoost'.

    Returns
    -------
    Outputs:

    filename_path : Path
        XXX
    """
    # For testing - comment out when not in use
    data_path = Path('/home/kcl/git/brain_age_clinical/data/dataset_harmonised.csv')
    # model_name = 'XGBoost'
    model_name = 'LinearSVR'
    # model_name = 'SVR_rbf'

    # Define input and output directories
    # data_dir = Path(PROJECT_ROOT / 'data')
    output_dir = Path(PROJECT_ROOT / 'output')
    model_dir = output_dir / model_name

    output_shap_dir = Path(model_dir / 'shapley_plots')
    output_shap_dir.mkdir(exist_ok=True)

    # Load data files containing Shapley values and age_predictions
    shap_values = pd.read_csv(model_dir / 'average_shap_values.csv')
    age_predictions = pd.read_csv(model_dir / 'age_predictions_validation.csv')

    data = pd.merge(shap_values, age_predictions[[
                    'id', 'Age', 'Gender', 'Diagn', 'Diagn_labels', 'dataset', 'scanner', 'analysis_categories']], on='id').set_index('id')

    # data.head()

    # Explode analysis_categories column to allowed comparison in pooled datasets
    data['analysis_categories'] = data['analysis_categories'].apply(ast.literal_eval)
    data_exploded = data.explode('analysis_categories')
    data_exploded['analysis_categories'] = data_exploded['analysis_categories'].astype(str)

    # Initialise a global list to store the top 15/10 regions from all diagnostic comparisons
    all_top_15_regions = []
    all_top_10_regions = []

    # Run analysis for all diagnostic groups
    # comparison_list = data_exploded.analysis_categories.unique()

    # Alternative comparison_list that ONLY includes comparisons that were significant in statistical_analysis.py
    comparison_list = ['17_comparison', '18_comparison', '2_comparison', '3_comparison', '11_comparison']

    for comparison in comparison_list:
        # Subset dataset to only this dataset
        subset_df = data_exploded[data_exploded.analysis_categories == comparison]

        # Subset by diagnostic group
        hc_df = subset_df[subset_df['Diagn'] == 1]
        patients_df = subset_df[subset_df['Diagn'] != 1]
        diagn_label = patients_df.Diagn_labels.unique()[0]

        print(f"{comparison} has {hc_df.shape[0]} HC and {patients_df.shape[0]} from diagn {diagn_label}")

        # Calculate mean Shapley value by diagnostic group
        hc_mean = hc_df[COLUMNS_NAME].mean()
        patients_mean = patients_df[COLUMNS_NAME].mean()

        # Calculate difference between patients and hc for each brain region
        diff = patients_mean - hc_mean

        # Get top 10 largest differences in absolute value
        top_10_regions = diff.abs().nlargest(10).index
        top_10_diff = diff[top_10_regions]

        # Create a single plot for this comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        top_10_diff.plot(kind='bar', ax=ax)

        # Add title and labels
        ax.set_title(
            f'Top 10 Regions with Largest Differences in Averaged Shapley Values between Patients and Controls - Datasets for {diagn_label}'
        )
        ax.set_xlabel('Brain Regions')
        ax.set_ylabel('Mean Difference (Patients - HC)')
        ax.set_xticklabels(top_10_regions, rotation=45)

        plt.tight_layout()

        # Save the plot for this dataset
        plot_filename = os.path.join(output_shap_dir, f'{comparison}_shapley_values.png')
        plt.savefig(plot_filename)
        plt.show()

        print(f"Plot for dataset {comparison} saved to", plot_filename)

        # Get top 15 largest differences in absolute values to add to the global list
        top_15_regions = diff.abs().nlargest(15).index
        all_top_15_regions.extend(top_15_regions)
        all_top_10_regions.extend(top_10_regions)

    # Significance testing for Shapley values of patients vs controls
    # Step 1: First decide which regions to test (multiple comparisons will need to be corrected for)
    # Step 2: Check for normality using Shapiro Wilks Test
    # Step 3: Conduct Mann-Whitney U test or OLS depending on normality assumption

    # Calculate the number of unique regions across all comparisons (Top 15 and Top 10 regions)
    unique_top_15_regions = set(all_top_15_regions)
    print(f"Number of unique regions across all comparisons (Top 15): {len(unique_top_15_regions)}")
    unique_top_10_regions = set(all_top_10_regions)
    print(f"Number of unique regions across all comparisons (Top 10): {len(unique_top_10_regions)}")

    # Initialise dictionaries to hold results from Shapiro Wilks, Mann-Whitney U, and OLS
    normality_violations = defaultdict(list)
    results_mwu = defaultdict(list)
    # results_ols = defaultdict(list)

    # Initialise list to store labels for heatmap figure
    comparison_labels = []

    for comparison in comparison_list:
        # Subset dataset to only this dataset
        subset_df = data_exploded[data_exploded.analysis_categories == comparison].copy()

        # Create column Diagn_binary where 1 in Diagn becomes 0 (HC), and anything else becomes 1 (patients) for OLS
        subset_df.loc[:, 'Diagn_binary'] = subset_df['Diagn'].apply(lambda x: 0 if x == 1 else 1)

        # Subset by diagnostic group
        hc_df = subset_df[subset_df['Diagn'] == 1]
        patients_df = subset_df[subset_df['Diagn'] != 1]
        diagn_label = patients_df.Diagn_labels.unique()[0]

        # Generate the new label for this comparison
        comparison_label = f"{diagn_label} vs HC"
        comparison_labels.append(comparison_label)

        print(f'Analysing SHAP comparison for: {comparison} --> {comparison_label}')

        # Either calculate for all 100 regions in COLUMNS_NAME or for the set of top 10 regions across all comparisons
        # for col in unique_top_10_regions:
        for col in COLUMNS_NAME:
            hc_shap = hc_df[col]
            patients_shap = patients_df[col]

            # Test for normality using Shapiro-Wilk Test
            hc_shapiro = stats.shapiro(hc_shap) if len(hc_shap) >= 3 else None
            sz_shapiro = stats.shapiro(patients_shap) if len(patients_shap) >= 3 else None
            if hc_shapiro and sz_shapiro:
                if hc_shapiro.pvalue < 0.05 or sz_shapiro.pvalue < 0.05:
                    normality_violations[col].append(comparison)
            # Note: The normality assumption is violated in the vast majority of regions, so Mann-Whitney U should be used

            # Conduct Mann-Whitney U Test
            mwu_res = stats.mannwhitneyu(hc_shap, patients_shap, alternative="two-sided")
            results_mwu[col].append(mwu_res.pvalue)

            # Conduct OLS Regression
            # This is done for comparison to Mann-Whitney U, as done by Ballester et al. (2023, Schizophrenia), but results are not shown
            # reg = smf.ols(f'Q("{col}") ~ Diagn_binary + Age + Gender', data=subset_df).fit()
            # ols_pvalue = reg.pvalues.get('Diagn_binary', np.nan)
            # results_ols[col].append(ols_pvalue)

    # Convert results to DataFrames
    results_mwu_df = pd.DataFrame.from_dict(results_mwu, orient='index', columns=comparison_labels)
    # results_ols_df = pd.DataFrame.from_dict(results_ols, orient='index', columns=comparison_labels)

    # Apply FDR correction
    mwu_p_values = results_mwu_df.to_numpy()
    # ols_p_values = results_ols_df.to_numpy()
    mwu_fdr_res = fdrcorrection(mwu_p_values.flatten())[1].reshape(mwu_p_values.shape)
    # ols_fdr_res = fdrcorrection(ols_p_values.flatten())[1].reshape(ols_p_values.shape)

    # Convert corrected results back to DataFrames
    mwu_results_fdr_df = pd.DataFrame(mwu_fdr_res, index=results_mwu_df.index, columns=comparison_labels)
    # ols_results_fdr_df = pd.DataFrame(ols_fdr_res, index=results_ols_df.index, columns=comparison_labels)

    # Map COLUMNS_NAME labels to new labels using FS2REGION - makes the figures more readable
    # First check for any NaN values
    updated_index = results_mwu_df.index.map(FS2REGION)

    # Check for NaN values in the mapping
    missing_labels = results_mwu_df.index[updated_index.isna()]

    # If there are missing labels, print them for manual checks
    if not missing_labels.empty:
        print("The following labels were not found in FS2REGION:")
        print(missing_labels.tolist())

    # Raise an error or warning if there are unmatched labels
    if not missing_labels.empty:
        raise ValueError("Some labels were not found in FS2REGION. Please update the mapping or check your data.")

    # Apply the mapping to update the index
    results_mwu_df.index = updated_index
    # results_mwu_df.index = results_mwu_df.index.map(FS2REGION)
    # results_ols_df.index = results_ols_df.index.map(FS2REGION)
    mwu_results_fdr_df.index = mwu_results_fdr_df.index.map(FS2REGION)
    # ols_results_fdr_df.index = ols_results_fdr_df.index.map(FS2REGION)

    # ---------------------------------------------------------------------------------------
    # Generate Heatmaps
    cmap = sns.color_palette("magma_r", as_cmap=True)

    # Heatmap for Mann-Whitney U
    print("Generating heatmap for Mann-Whitney U test")

    # Determine if all regions or only those with significant differences for at least one diagnosis should be plotted
    plot_sig_only = False

    # Filter only regions that are significant (any comparison < 0.05)
    sig_mask = (mwu_results_fdr_df < 0.05).any(axis=1)
    mwu_results_fdr_sig_df = mwu_results_fdr_df[sig_mask]

    # Shorten x-axis labels for cleaner plots
    short_label_map = {
        "Alzheimer's Disease vs HC": "AD vs HC",
        "Mild Cognitive Impairment vs HC": "MCI vs HC",
        "Schizophrenia vs HC": "SZ vs HC",
        "Bipolar Disorder vs HC": "BS vs HC",
        "Autism Spectrum Disorder vs HC": "ASD vs HC",
    }

    # Apply short labels
    short_labels = [short_label_map.get(label, label) for label in comparison_labels]
    mwu_results_fdr_df.columns = short_labels
    mwu_results_fdr_sig_df.columns = short_labels

    # Optionally filter to only significant rows
    if plot_sig_only:
        # Filter only regions that are significant (any comparison < 0.05)
        sig_mask = (mwu_results_fdr_df < 0.05).any(axis=1)
        mwu_results_fdr_sig_df = mwu_results_fdr_df[sig_mask]

        if mwu_results_fdr_sig_df.empty:
            print("No significant regions found with FDR < 0.05. Skipping heatmap generation.")

        plot_df = mwu_results_fdr_sig_df
        suffix = "sigonly"
    else:
        plot_df = mwu_results_fdr_df
        suffix = "allregions"

    n_rows = plot_df.shape[0]
    n_cols = plot_df.shape[1]
    vmin, vmax = 0, 0.05  # Shared color scale

    if n_rows < 50:
        # Single-column version
        fig, ax = plt.subplots(figsize=(n_cols * 2.5, n_rows * 0.35 + 1.5))
        sns.heatmap(plot_df, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                    annot=True, annot_kws={'size': 8}, cbar_kws={'label': 'FDR-adjusted p-value'})

        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f"FDR-adjusted p-values ({'Significant Only' if plot_sig_only else 'All Regions'})")

        plt.tight_layout()
        plt.savefig(output_shap_dir / f"heatmap_mwu_{suffix}_singlecolumn.png", bbox_inches='tight')
        plt.show()
    else:
        # Two-column version
        half = n_rows // 2
        df_left = plot_df.iloc[:half, :]
        df_right = plot_df.iloc[half:, :]

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(n_cols * 2.5, half * 0.35 + 1.5),
                                 sharex=True, sharey=False, gridspec_kw={'width_ratios': [1, 1]})
        cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])

        sns.heatmap(df_left, ax=axes[0], cmap=cmap, vmin=vmin, vmax=vmax,
                    annot=True, annot_kws={'size': 7}, cbar=False)
        sns.heatmap(df_right, ax=axes[1], cmap=cmap, vmin=vmin, vmax=vmax,
                    annot=True, annot_kws={'size': 7}, cbar=True, cbar_ax=cbar_ax)

        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xlabel('')
            ax.set_ylabel('')

        # axes[0].set_title(f"Regions 1–{half}")
        # axes[1].set_title(f"Regions {half + 1}–{n_rows}")

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        # plt.suptitle(f"FDR-adjusted p-values from Mann-Whitney U Test ({'Significant Only' if plot_sig_only else 'All Regions'})",
        #              fontsize=14, y=1.02)

        plt.savefig(output_shap_dir / f"heatmap_mwu_{suffix}_splitcolumns.png", bbox_inches='tight')
        plt.show()

    # # Heatmap for OLS
    # print("Generating heatmap for OLS")
    #
    # # Dynamic figure size
    # n_rows = ols_results_fdr_df.shape[0]
    # n_cols = ols_results_fdr_df.shape[1]
    # fig_width = n_cols * 2
    # fig_height = n_rows * 0.5
    # plt.figure(figsize=(fig_width, fig_height))
    #
    # sns.heatmap(ols_results_fdr_df, annot=True, annot_kws={'size': 8}, vmin=0, vmax=0.05, cmap=cmap, cbar_kws={'label': 'FDR-adjusted p-value'})
    # plt.title("OLS Regression (FDR-corrected)")
    # plt.xticks(rotation=45, ha="right", fontsize=10)
    # plt.yticks(rotation=0, ha="right", fontsize=10)
    # plt.tight_layout()
    # plt.savefig(output_shap_dir / "heatmap_ols_allregions_sigonly.png")
    # # plt.savefig(output_shap_dir / "heatmap_ols_allregions.png")
    # # plt.savefig(output_shap_dir / "heatmap_ols_top10.png")
    # plt.show()

    # Report regions with normality violations
    print("\nNormality violations detected in the following regions/datasets:")
    for region, datasets_violated in normality_violations.items():
        print(f"{region}: {', '.join(datasets_violated)}")


if __name__ == '__main__':
    run_shap_analysis()
