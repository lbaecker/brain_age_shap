"""Script to apply trained models to unseen data to assess generalisation performance.

The script loops over the 100 models created in from train_model.py, loads their scaler and regressors, applies
them to the data defined as 'validation_ids' in fetch_data.py and saves all predictions per subjects
in age_predictions_validation.csv.

It calculates the Brain Age Gap (BAG) (predicted age - chronological age), and
Brain Age Gap Estimate Residualised (BrainAGER) [1], which is thought to minimise the age bias.

References:
[1] Le TT, Kuplicki RT, McKinney BA, Yeh H-W, Thompson WK, Paulus MP and Tulsa 1000 Investigators (2018)
A Nonlinear Simulation Framework Supports Adjusting for Age When Analyzing BrainAGE.
Front. Aging Neurosci. 10:317. doi: 10.3389/fnagi.2018.00317
"""
import random
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
import ast

from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import os
import numpy as np
from joblib import load, Parallel, delayed
from scipy import stats
import shap

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

from src.definitions import COLUMNS_NAME

PROJECT_ROOT = Path.cwd()


def get_box_plot(df, metric='brain_age_gap', grouping_var='scanner', output_path=None, filename='box_plot', x_axis_limit=False):
    """ Box-and-whisker plot for the chosen deviation metric.

    Parameters
    ----------
    df: NDFrame
        Dataframe containing the metric to plot.
    metric: str, default='brain_age_gap'
        Name of the column with the chosen metric to plot. Default is the brain age gap.
    grouping_var: str, default='scanner'
        Name of variable that determines how the plot is displayed. If 'scanner' the overall metric is displayed
        separately for each scanner.
    output_path: str, default=None
        Name of directory where figures will be saved. If None, the figure is not saved.
    filename: str, default='box_plot'
        File name of the resulting figure saved in output_path.
    x_axis_limit: boolean, default=False
        Restrict the scale of the x axis to align between different models.
        Warning: This may exclude some data points.

    Returns
    ---------
    g: Matplotlib figure
        Box-and-whisker plot for the chosen deviation metric.
    image_path : str
        Path to where the plot is saved.
    """
    names = df[grouping_var].unique()
    names = np.sort(names)
    n_cols = len(names)

    box_fig, axes = plt.subplots(n_cols, 1, figsize=(10, 15), sharex='col')

    if grouping_var != 'Diagn':
        for name, ax in zip(names, axes.flatten()):
            sns.boxplot(data=df[df[grouping_var] == name], x=metric, y='Diagn', orient='h', ax=ax).set_title(name)
            # fig.set_title(name)
    else:
        sns.boxplot(data=df, x=metric, y='Diagn', orient='h').set_title(grouping_var)

    # OPTIONAL: Set x-axis limits
    if x_axis_limit is True:
        plt.xlim(-60, 60)

    plt.tight_layout()
    # plt.show()

    image_path = f'{output_path}/{filename}.png'
    if output_path is not None:
        plt.savefig(image_path)

    # box_fig.savefig(Path(output_path / 'box_plot.png'))
    plt.close()

    return image_path


def get_box_plot_hc(df, metric='brain_age_gap', output_path=None, filename='box_plot_HC', x_axis_limit=False):
    """Plot the brain age gaps of healthy controls to assess model generalisability.

    Parameters
    ----------
    df: NDFrame
        Dataframe containing the metric to plot.
    metric: str, default='brain_age_gap'
        Name of the column with the chosen metric to plot. Could be 'brain_age_gap' or 'BrainAGER'.
    output_path: str, default=None
        Name of directory where figures will be saved. If None, the figure is not saved.
    filename: str, default='box_plot_HC'
        File name of the resulting figure saved in output_path.
    x_axis_limit: boolean, default=False
        Restrict the scale of the x axis to align between different models.
        Warning: This may exclude some data points.

    Returns
    ---------
    g: Matplotlib figure
        Box-and-whisker plot for the chosen deviation metric.
    image_path : str
        Path to where the plot is saved.
    """
    # Prep data
    plot_data = df[df['Diagn'] == 1][['dataset', metric]]

    # Plot
    sns.boxplot(data=plot_data, x=metric, y='dataset', orient='h').set_title('Performance in HC only')

    # OPTIONAL: Set x-axis limits
    if x_axis_limit is True:
        plt.xlim(-60, 60)

    # Save plot
    image_path = f'{output_path}/{filename}.png'
    if output_path is not None:
        plt.savefig(image_path)

    plt.close()
    return image_path


def prep_data_plot(df, metric='deviations', grouping_var='scanner'):
    """Prepare data for ridge plot.

    Parameters
    ----------
    df: NDFrame
        Dataframe containing the metric to plot.
    metric: str, default='deviations'
        Name of the column with the chosen metric to plot.
    grouping_var: str, default='scanner'
        Name of variable that determines how the plot is displayed. If 'scanner' the overall metric is displayed
        separately for each scanner; if 'group_scanner' the curves for each scanner*diagnostic group are shown
        separately instead.

    Returns
    ---------
    df: NDFrame
        Dataframe containing the metric to plot.
    """
    # x are the deviation scores for HC and y the deviation scores for clinical subjects
    x = df.loc[df['Diagn'] == 1][metric]
    y = df.loc[df['Diagn'] != 1][metric]
    g = df[grouping_var]
    df_out = pd.DataFrame(dict(x=x, y=y, g=g))
    df_out.columns = ['HC', 'P', grouping_var]
    df_out = df_out.sort_values(grouping_var)
    return df_out


def get_ridge_plot(df, metric='deviations', grouping_var='scanner', output_path=None, filename='ridge_plot'):
    """Ridge plot for the chosen deviation metric.

    Parameters
    ----------
    df: NDFrame
        Dataframe containing the metric to plot.
    metric: str
        Name of the column with the chosen metric to plot.
    grouping_var: str
        Name of variable that determines how the plot is displayed. If 'scanner' the overall metric is displayed
        separately for each scanner.
    output_path: str, default=output_path
        Name of directory where figures will be saved.

    Returns
    ---------
    g: Seaborn FacetGrid
        Ridge plot for the chosen deviation metric.
    """
    data = prep_data_plot(df, metric, grouping_var)

    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Note: use bw_adjust to adjust smoothing
    g = sns.FacetGrid(data, row=grouping_var, hue=grouping_var, aspect=10, height=3)
    # g = sns.FacetGrid(data, row=grouping_var, hue=grouping_var, aspect=15, height=1.5)
    g.map(sns.kdeplot, 'HC', clip_on=False, fill=True, alpha=0.8, lw=0.5, bw_method='scott', bw_adjust=1, color="grey")
    g.map(sns.kdeplot, "P", clip_on=False, fill=True, alpha=0.8, lw=0.5, bw_method='scott', bw_adjust=1, color="r")

    g.map(plt.axhline, y=0, lw=2, clip_on=False, color='grey')
    g.map(plt.axvline, x=0, c='grey')

    def label(color, label):
        ax = plt.gca()
        ax.text(0.7, 0.2, label, fontweight="bold", ha="left",
                va="center", fontsize=16, transform=ax.transAxes)

    g.map(label)
    g.set_xlabels(metric)
    g.fig.subplots_adjust(hspace=-.2)
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    g.fig.suptitle('')

    image_path = f'{output_path}/{filename}.png'
    if output_path is not None:
        plt.savefig(image_path)

    # box_fig.savefig(Path(output_path / 'box_plot.png'))

    return image_path


def get_brainager(age, predicted_age):
    """Calculates BrainAGER, the residual error of the regression of chronological and predicted brain age"""
    age = sm.add_constant(age)
    model = sm.OLS(predicted_age, age)
    results = model.fit()
    residuals = results.resid

    return residuals


def calculate_bag_patients_hc(df, metric='brain_age_gap', grouping_var='dataset', output_path=None, filename='difference_BAG_patients_HC.csv'):
    """Calculates the difference in mean brain age gap between patients and healthy controls within each scanner.

    Parameters
    ----------
    df: NDFrame
        Dataframe containing the metric to calculate.
    metric: str, default='brain_age_gap'
        Name of the column with the chosen metric to calculate.
    grouping_var: str, default='dataset'
        Name of variable that determines how the calculations are grouped. Default is 'dataset', could also be 'scanner'.
    output_path: PosixPath, default=None
        Path of directory where results will be saved. If None, the results are not saved.
    filename: str, default='difference_BAG_patients_HC.csv'
        File name of the results to be saved in output_path.

    Returns
    ---------
    merged: NDFrame
        Dataframe containing mean brain age gap between each diagnosis and healthy controls per dataset.
    mean_brain_age_gap_1 = NDFrame
        Dataframe containing mean brain age gap for healthy controls per scanner.
    """
    # Filter out rows where Diagn=1
    df_non_1 = df[df['Diagn'] != 1]

    # Calculate the mean brain_age_gap for Diagn=1
    mean_brain_age_gap_1 = df[df['Diagn'] == 1].groupby(grouping_var)[metric].mean()

    # Group by grouping_var and 'Diagn', calculate the mean of metric
    grouped = df_non_1.groupby([grouping_var, 'Diagn'])[metric].mean().reset_index()

    # Merge with the mean brain_age_gap for Diagn=1
    merged = pd.merge(grouped, mean_brain_age_gap_1, how='left', on=grouping_var, suffixes=('_non_1', '_1'))

    # Calculate the difference between mean brain_age_gap for each non-1 Diagn and Diagn=1
    var1_name = metric + '_non_1'
    var2_name = metric + '_1'
    merged['difference'] = merged[var1_name] - merged[var2_name]

    if output_path is not None:
        merged.to_csv(output_path / filename)

    return merged, mean_brain_age_gap_1


def apply_model(data_path, model_name='SVR_rbf', n_repetitions=10, n_folds=10, calculate_shap=True):
    """Apply brain age model trained in train_model.py.

    Parameters
    ----------
    data_path : Path
        Path to the dataset created in run_combat.py.
    model_name : str, default='SVR_rbf'
        Model type for brain age prediction. Could be 'LinearSVR', 'SVR_rbf' or 'XGBoost'.
    n_repetitions : int, default=10
        Number of repetitions for brain age model.
    n_folds : int, default=10
        Number of folds for cross-validation of brain age model.
    calculate_shap : bool, default=True
        Whether Shapley values for regional contributions should be calculated.

    Returns
    -------
    Outputs:

    output_filepath : Path
        Path where output predictions for patients and controls are saved.
    """
    # Define input and output directories
    # data_dir = Path(PROJECT_ROOT / 'data')
    output_dir = Path(PROJECT_ROOT / 'output')

    model_dir = output_dir / model_name
    training_cv_dir = model_dir / 'cv'
    validation_output_dir = model_dir / 'validation'
    validation_output_dir.mkdir(exist_ok=True)
    val_cv_dir = validation_output_dir / 'cv'
    val_cv_dir.mkdir(exist_ok=True)

    model_output_dir = model_dir

    # Load dataset file containing demographic as well as FreeSurfer data
    dataset = pd.read_csv(data_path)

    # Use validation data only (patients and controls)
    dataset_val = dataset[dataset.groups == 'validation_ids']

    # ----------------------------------------------------------------------------------------
    print('************** APPLYING BRAIN AGE MODEL **************')

    # Initialise random seed
    np.random.seed(42)
    random.seed(42)

    # Normalise regional volumes by total intracranial volume (tiv)
    regions = dataset_val[COLUMNS_NAME].values

    tiv = dataset_val.EstimatedTotalIntraCranialVol.values[:, np.newaxis]

    regions_norm = np.true_divide(regions, tiv)
    age = dataset_val['Age'].values

    # Create dataframe to hold actual and predicted ages
    age_predictions = dataset_val[['id', 'Age', 'Gender', 'Diagn',
                                   'dataset', 'analysis_categories', 'Diagn_labels', 'scanner']]
    age_predictions = age_predictions.set_index('id')

    # Create empty dict to accumulate Shapley values per subject across iterations
    subjects = age_predictions.index.to_list()
    # shap_values_dict = {subject: np.zeros(len(COLUMNS_NAME)) for subject in subjects}
    shap_values_dict_list = []

    # Function to compute SHAP values on a specific GPU
    # Note: This is for the models that are not LinearSVR; not functional at the moment
    def compute_shap_values(i, x_test_chunk, background, gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Convert memmap to ndarray if necessary
        if isinstance(x_test_chunk, np.memmap):
            x_test_chunk = np.array(x_test_chunk)
        if isinstance(background, np.memmap):
            background = np.array(background)

        # Create a function to predict with the trained model
        def model_predict(X):
            return model.predict(X)

        explainer = shap.KernelExplainer(model_predict, background)
        shap_values_chunk = explainer.shap_values(x_test_chunk)
        return shap_values_chunk

    # Loop over the model iterations and apply them to the data
    for i_repetition in range(n_repetitions):
        for i_fold in range(n_folds):
            # Load model and scaler
            prefix = f'{i_repetition:02d}_{i_fold:02d}'

            model = load(training_cv_dir / f'{prefix}_regressor.joblib')
            scaler = load(training_cv_dir / f'{prefix}_scaler.joblib')

            # Use RobustScaler to transform testing data
            x_test = scaler.transform(regions_norm)

            # Apply model to scaled data
            predictions = model.predict(x_test)

            # ----------------------------------------
            # Assess regional contributions using Shapley values
            # Please note: LinearSVR can be handled by the inbuilt functionality of the shap library;
            # SVR with a rbf kernel requires KernelExplainer (instead of Explainer) for a model-agnostic approach.
            # XGBoost requires TreeExplainer (instead of Explainer) for a model-agnostic approach - not implemented yet.

            if calculate_shap is True:
                if model_name in ['LinearSVR']:
                    # Create SHAP explainer
                    explainer = shap.Explainer(model, x_test)

                    # Compute Shapley values
                    shap_values = explainer(x_test)
                    # Note: The shap_values object contains several attributes that provide different pieces of information
                    # about the model's predictions (.values, .base_values, .data)
                    # .values contains the Shapley values for each feature for each instance (number_of_instances, number_of_features)
                    # These values represent the contribution of each feature to the difference between the model's prediction
                    # for a given instance and the baseline prediction (the average prediction over the training set or
                    # a specified reference value).

                    # Store Shapley values
                    # Convert Shapley values to a dictionary for easier aggregation
                    # print(shap_values.base_values)
                    shap_values_dict = {subjects[i]: shap_values[i].values for i in range(len(subjects))}

                    # Append the Shapley values dictionary to the list
                    shap_values_dict_list.append(shap_values_dict)
                elif model_name == 'SVR_rbf':
                    # Number of GPUs available
                    num_gpus = 2

                    # Split the x_test data into chunks for each GPU
                    x_test_chunks = np.array_split(x_test, num_gpus)

                    # Use a subset of training data as the background dataset for the KernelExplainer
                    background = x_test[np.random.choice(x_test.shape[0], 50, replace=False)]

                    # Ensure background is a numpy array (not memmap)
                    if isinstance(background, np.memmap):
                        background = np.array(background)

                    # Parallel SHAP value computation
                    results = Parallel(n_jobs=num_gpus)(delayed(compute_shap_values)(
                        i, x_test_chunks[i], background, i) for i in range(num_gpus))

                    # Combine the SHAP values from all chunks
                    shap_values = np.concatenate(results, axis=0)

                    # Store Shapley values
                    shap_values_dict = {subjects[i]: shap_values[i] for i in range(len(subjects))}
                    shap_values_dict_list.append(shap_values_dict)
                else:
                    print("Issue with model_name in the Shapley calculations")

            # ----------------------------------------
            # Assess model performance

            # Note: This is patients and HC mixed
            mae = mean_absolute_error(age, predictions)
            rmse = sqrt(mean_squared_error(age, predictions))
            r, _ = stats.pearsonr(age, predictions)
            r2 = r2_score(age, predictions)
            age_error_corr, _ = stats.spearmanr((predictions - age), age)

            # ----------------------------------------
            # Save outputs

            # Save prediction per model in df
            age_predictions[f'Prediction {i_repetition:02d}_{i_fold:02d}'] = predictions

            # Save model scores
            scores_array = np.array([r, r2, mae, rmse, age_error_corr])
            np.save(val_cv_dir / f'{prefix}_scores.npy', scores_array)

    # ---------------------------------------------------------
    # Calculate mean Shapley values for all features
    if calculate_shap is True:
        # Initialize a dictionary to store averaged Shapley values
        avg_shap_values_dict = {subject: np.zeros(len(COLUMNS_NAME)) for subject in subjects}

        # Aggregate Shapley values across all models
        for shap_values_dict in shap_values_dict_list:
            for subject in subjects:
                avg_shap_values_dict[subject] += shap_values_dict[subject]

        # Average the Shapley values across all models
        num_models = n_repetitions * n_folds
        for subject in avg_shap_values_dict:
            avg_shap_values_dict[subject] /= num_models

        # Create DataFrame
        df_avg_shap_values = pd.DataFrame.from_dict(avg_shap_values_dict, orient='index', columns=COLUMNS_NAME)
        df_avg_shap_values = df_avg_shap_values.rename_axis('id')

        # Save the DataFrame to a CSV file
        df_avg_shap_values.to_csv(model_output_dir / 'average_shap_values.csv', index=True)

    # ---------------------------------------------------------
    # Get mean age predictions across repetitions
    max_repetition_col_str = f'Prediction {(n_repetitions - 1):02d}_{(n_folds - 1):02d}'
    repetition_cols = age_predictions.loc[:, 'Prediction 00_00': max_repetition_col_str]
    age_predictions['prediction_mean'] = repetition_cols.mean(axis=1)

    # Calculate brain age gap
    age_predictions['brain_age_gap'] = age_predictions['prediction_mean'] - age_predictions['Age']

    age_predictions['BrainAGER'] = get_brainager(age_predictions['Age'],
                                                 age_predictions['prediction_mean'])

    # Save age predictions
    output_filename = 'age_predictions_validation.csv'
    output_filepath = model_output_dir / output_filename
    age_predictions.to_csv(output_filepath)

    # -------------------------------------------------------------------------------------------
    # Get generalisation performance in healthy controls

    # Subset data to HC only
    hc_age_predictions = age_predictions[age_predictions.Diagn == 1]

    print(age_predictions.shape[0], hc_age_predictions.shape[0])
    sample_size = hc_age_predictions.shape[0]

    hc_predictions = hc_age_predictions['prediction_mean']
    hc_age = hc_age_predictions['Age']

    hc_mean_brainage_gap = (hc_predictions - hc_age).mean()
    hc_mae = mean_absolute_error(hc_age, hc_predictions)
    hc_rmse = sqrt(mean_squared_error(hc_age, hc_predictions))
    hc_r, _ = stats.pearsonr(hc_age, hc_predictions)
    hc_r2 = r2_score(hc_age, hc_predictions)
    hc_age_error_corr, _ = stats.spearmanr((hc_predictions - hc_age), hc_age)

    print(f'Generalisation performance in HC in the independent dataset:\n'
          f'Sample size of HC across datasets: {sample_size}\n'
          f'MAE: {hc_mae:0.3f}\n'
          f'RMSE: {hc_rmse:0.3f}\n'
          f'Mean brain age gap: {hc_mean_brainage_gap:0.3f}\n'
          f'r: {hc_r:0.3f}\n'
          f'R2: {hc_r2:0.3f}\n'
          f'CORR: {hc_age_error_corr:0.3f}\n')
    generalisation_dict = {
        'Metric': [
            'Sample size of HC across datasets',
            'MAE',
            'RMSE',
            'Mean brain age gap',
            'r',
            'R2',
            'CORR'
        ],
        'Value': [
            sample_size,
            hc_mae,
            hc_rmse,
            hc_mean_brainage_gap,
            hc_r,
            hc_r2,
            hc_age_error_corr
        ]
    }

    generalisation_df = pd.DataFrame(generalisation_dict)
    generalisation_df.to_csv(model_output_dir / 'generalisation_performance.csv', index=False)

    # -------------------------------------------------------------------------------------------
    # Analysis

    # Assess age bias
    r, p_value = stats.pearsonr(age_predictions['Age'], age_predictions['brain_age_gap'])
    print("Pearson's r:", r)
    print("p-value:", p_value)

    # Analysis of patients vs controls
    # Explode analysis_categories column to allowed comparison in pooled datasets
    age_predictions['analysis_categories'] = age_predictions['analysis_categories'].apply(ast.literal_eval)
    age_predictions_exploded = age_predictions.explode('analysis_categories')
    age_predictions_exploded['analysis_categories'] = age_predictions_exploded['analysis_categories'].astype(str)
    print(age_predictions.shape, age_predictions_exploded.shape)

    # Separate performance by dataset or scanner
    # Get MAE from validation set per diagnosis (within each scanner)
    mean_bag_dict = {}
    for i_comparison in age_predictions_exploded.analysis_categories.unique():
        dataset = age_predictions_exploded[age_predictions_exploded.analysis_categories == i_comparison]
        # print(i_comparison)
        for diagn_name in dataset.Diagn_labels.unique():
            # print(diagn_name)
            var_name = str(i_comparison) + '_' + str(diagn_name) + "_mean_BAG"
            mean_bag = dataset[dataset.Diagn_labels == diagn_name]["brain_age_gap"].mean()
            # mean_bag = dataset[dataset.Diagn_labels == diagn_name]["brain_age_gap"].mean().round(3)
            mean_bag_dict[var_name] = mean_bag

    mean_bag_df = pd.DataFrame(mean_bag_dict, index=[1])
    mean_bag_df = mean_bag_df.T
    mean_bag_df.to_csv(model_output_dir / 'mean_bag_df.csv')

    # Within each dataset, calculate the difference in mean brain age gap between HC and each diagnosis
    bag_differences, mean_brain_age_gap_1 = calculate_bag_patients_hc(
        age_predictions_exploded, metric='brain_age_gap', grouping_var='analysis_categories', output_path=model_output_dir, filename='difference_BAG_patients_HC.csv')

    # Do the same for BrainAGER
    bag_differences_brainager, mean_brainager_1 = calculate_bag_patients_hc(
        age_predictions_exploded, metric='BrainAGER', grouping_var='analysis_categories', output_path=model_output_dir, filename='difference_BrainAGER_patients_HC.csv')

    # -------------------------------------------------------------------------------------------
    # Generate plots

    # Box plot of brain age gap of HC in independent datasets only to assess generalisability
    brain_age_gap_hc = get_box_plot_hc(age_predictions, metric='brain_age_gap',
                                       output_path=model_output_dir, filename='box_plot_HC_brain_age_gap',
                                       x_axis_limit=True)
    brainager_hc = get_box_plot_hc(age_predictions, metric='BrainAGER',
                                   output_path=model_output_dir, filename='box_plot_HC_BrainAGER',
                                   x_axis_limit=True)

    # Plot these results of diagnosis by dataset
    box_plot_byscanner = get_box_plot(age_predictions, metric='brain_age_gap',
                                      grouping_var='dataset', output_path=model_output_dir, filename='box_plot_agegap',
                                      x_axis_limit=True)
    box_plot_bycomparison = get_box_plot(age_predictions_exploded, metric='brain_age_gap',
                                         grouping_var='analysis_categories', output_path=model_output_dir, filename='box_plot_agegap_analysis_categories',
                                         x_axis_limit=True)
    box_plot_byscanner_brainager = get_box_plot(age_predictions, metric='BrainAGER',
                                                grouping_var='dataset', output_path=model_output_dir,
                                                filename='box_plot_brainAGER',
                                                x_axis_limit=True)

    return output_filepath


if __name__ == '__main__':
    apply_model()
