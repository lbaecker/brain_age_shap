"""Script for posthoc statistical analysis and figures for manuscript.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path
import ast
from statsmodels.stats.multitest import multipletests
import numpy as np

PROJECT_ROOT = Path.cwd()


def analyse_brainage(data_path, model_name='LinearSVR'):
    """Analyse brain age of patients vs controls, controlling for age and gender.

    Statistical analysis of mean brain age gap between patients and control
    Linear regression, controlling for age and gender, for each diagnostic comparison
    Dependent variable = brain age gap, independent variable = diagnosis

    Parameters
    ----------
    data_path : Path
        Path to the dataset created in apply_model.py containing patient and HC predictions.
    model_name : str, default='LinearSVR'
        Model type for brain age prediction. Could be 'LinearSVR', 'SVR_rbf' or 'XGBoost'.

    Returns
    -------
    Outputs:

    filename_path : Path
        Path where regression output for all analyses are saved.
    """
    # Load validation data (incl. clinical subjects)
    data = pd.read_csv(data_path)

    output_dir = Path(PROJECT_ROOT / 'output' / model_name / 'analyse_brainage')
    output_dir.mkdir(exist_ok=True)

    # Explode analysis_categories column to allowed comparison in pooled datasets
    data['analysis_categories'] = data['analysis_categories'].apply(ast.literal_eval)
    data_exploded = data.explode('analysis_categories')
    data_exploded['analysis_categories'] = data_exploded['analysis_categories'].astype(str)

    # Initialize an empty list to collect results
    results_list = []

    comparison_list = data_exploded.analysis_categories.unique()

    for comparison in comparison_list:
        # Subset dataset to only this dataset
        subset_df = data_exploded[data_exploded.analysis_categories == comparison]

        # Get diagnosis name
        patients_df = subset_df[subset_df['Diagn'] != 1]
        diagn_label = patients_df.Diagn_labels.unique()[0]
        print('Running OLS analysis for brain age gap in: ', diagn_label)

        subset = subset_df.copy()

        # Create column Diagn_binary where 1 in Diagn becomes 0, and anything else becomes 1
        subset.loc[:, 'Diagn_binary'] = subset['Diagn'].apply(lambda x: 0 if x == 1 else 1)

        # Optional to check it was converted correctly
        # subset.Diagn.value_counts()
        # subset.Diagn_binary.value_counts()
        #
        # Conduct OLS regression
        reg = smf.ols("brain_age_gap ~ Diagn_binary + Age + Gender", data=subset).fit()
        # reg.summary()

        # Extract specific parameters
        coeff_diagn = reg.params['Diagn_binary'].round(2)
        pvalue_diagn = reg.pvalues['Diagn_binary'].round(5)
        conf_int = reg.conf_int()
        lower_ci_diagn = conf_int.loc['Diagn_binary', 0].round(5)  # Lower bound
        upper_ci_diagn = conf_int.loc['Diagn_binary', 1].round(5)  # Upper bound

        # Append the results to the list
        results_list.append({
            'Analysis': comparison,
            'Diagn': diagn_label,
            'Coef': coeff_diagn,
            'pval': pvalue_diagn,
            'lower_CI': lower_ci_diagn,
            'upper_CI': upper_ci_diagn
        })

    # Create DataFrame from the results list
    df_results = pd.DataFrame(results_list)

    df_results_sorted = df_results.sort_values(by='Diagn')

    # Apply Benjamini-Hochberg FDR correction for multiple comparisons to p-values
    pvals = df_results_sorted['pval'].values
    fdr_results = multipletests(pvals, alpha=0.05, method='fdr_bh')
    df_results_sorted['pval_corrected'] = fdr_results[1]
    df_results_sorted['significant'] = fdr_results[0]  # True if significant

    # Print summary of significant results
    print("\nSummary of significant results after FDR correction:")
    print(df_results_sorted[df_results_sorted['significant']])

    # Save results to CSV
    output_filename = 'ols_results_with_fdr.csv'
    ols_output_filepath = output_dir / output_filename
    df_results_sorted.to_csv(ols_output_filepath, index=False)

    # ---------------------------------------------------------------------------------
    # Get final boxplot of patient comparisons
    # This is the same as the output of apply_model, just labeled differently
    df = data_exploded
    metric = 'brain_age_gap'
    grouping_var = 'analysis_categories'
    output_path = output_dir
    filename = 'box_plot_agegap_analysis_categories_v2'
    x_axis_limit = True

    names = df[grouping_var].unique()
    names = np.sort(names)
    n_cols = len(names)

    # box_fig = plt.figure(figsize=(20, 20))
    box_fig, axes = plt.subplots(n_cols, 1, figsize=(10, 15), sharex='col')

    if grouping_var != 'Diagn_labels':
        for name, ax in zip(names, axes.flatten()):
            subset_df = df[df[grouping_var] == name]
            sns.boxplot(data=subset_df, x=metric, y='Diagn_labels', orient='h', ax=ax).set_title(name)

            # Determine and set a title for the subblot
            patients_df = subset_df[subset_df['Diagn'] != 1]
            diagn_name = patients_df.Diagn_labels.unique()[0]
            title_name = f"{diagn_name} vs HC"
            ax.set_title(title_name)

            # Remove y-axis labels
            ax.set_ylabel(None)
    else:
        sns.boxplot(data=df, x=metric, y='Diagn_labels', orient='h').set_title(grouping_var)

    # OPTIONAL: Set x-axis limits
    if x_axis_limit is True:
        plt.xlim(-60, 60)

    # Set x-axis label only once at the bottom
    plt.xlabel('Mean brain age gap')

    plt.tight_layout()
    # plt.show()

    boxplot_image_path = f'{output_path}/{filename}.png'
    if output_path is not None:
        plt.savefig(boxplot_image_path)

    # box_fig.savefig(Path(output_path / 'box_plot.png'))
    plt.close()

    return ols_output_filepath, boxplot_image_path


if __name__ == '__main__':
    analyse_brainage()
