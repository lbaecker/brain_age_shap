"""Script to train brain age model using SVR.

This script is based on the previously published code in Baecker et al. 2021 (Human Brain Mapping).

We trained the Support Vector Machines (SVMs) [1] in 10 repetitions of
10 stratified k-fold cross-validation (CV) (stratified by age).
The hyperparameter tuning was performed in an automatic way using
nested CV.

References
----------
[1] - Cortes, Corinna, and Vladimir Vapnik. "Support-vector networks."
Machine learning 20.3 (1995): 273-297.
"""
import pandas as pd
import random
from math import sqrt
import matplotlib.pyplot as plt
from pathlib import Path

import shap
import numpy as np
from joblib import dump
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVR, SVR
from xgboost import XGBRegressor

from src.definitions import COLUMNS_NAME

import warnings
# from pandas.errors import PerformanceWarning

# warnings.simplefilter("ignore", PerformanceWarning)
warnings.simplefilter("ignore", FutureWarning)

PROJECT_ROOT = Path.cwd()


def train_model(data_path, model_name='SVR', n_repetitions=10, n_folds=10):
    """Train brain age model using linear SVR.

    Parameters
    ----------
    data_path : Path
        Path to the dataset created in run_combat.py.
    model_name : str, default='SVR'
        Model type for brain age prediction. Could be 'SVR' or 'XGBoost'
    n_repetitions : int, default=10
        Number of repetitions for brain age model.
    n_folds : int, default=10
        Number of folds for cross-validation of brain age model.

    Returns
    -------
    Outputs:

    filename_path : Path
        XXX
    """
    # Define input and output directories
    # data_dir = Path(PROJECT_ROOT / 'data')
    output_dir = Path(PROJECT_ROOT / 'output')
    output_dir.mkdir(exist_ok=True)

    model_dir = output_dir / model_name
    model_dir.mkdir(exist_ok=True)
    cv_dir = model_dir / 'cv'
    cv_dir.mkdir(exist_ok=True)

    # Load dataset file containing demographic as well as FreeSurfer data
    dataset = pd.read_csv(data_path)

    # Use training data only
    dataset_train = dataset[dataset.groups == 'train_ids']
    # print(dataset_train.shape[0])

    # ----------------------------------------------------------------------------------------
    print('************** TRAINING BRAIN AGE MODEL **************')

    # Initialise random seed
    np.random.seed(42)
    random.seed(42)

    # Normalise regional volumes by total intracranial volume (tiv)
    regions = dataset_train[COLUMNS_NAME].values

    tiv = dataset_train.EstimatedTotalIntraCranialVol.values[:, np.newaxis]

    regions_norm = np.true_divide(regions, tiv)
    age = dataset_train['Age'].values

    # dataset_train.Age.value_counts()

    # CV variables
    cv_mean_brainage_gap = []
    cv_r = []
    cv_r2 = []
    cv_mae = []
    cv_rmse = []
    cv_age_error_corr = []

    # Initialise lists to hold model coefficients
    # These can be compared to the train SHAP values
    coefficients_list = []

    # Create DataFrame to hold actual and predicted ages
    age_predictions = dataset_train[['id', 'Age']]
    age_predictions = age_predictions.set_index('id')

    # Initialize a DataFrame to store Shapley values
    shap_values_df = pd.DataFrame(0, index=dataset_train.id, columns=COLUMNS_NAME)

    # Specify number of nested folds
    n_nested_folds = 5

    for i_repetition in range(n_repetitions):
        # Create new empty column in age_predictions df to save age predictions of this repetition
        repetition_column_name = f'Prediction repetition {i_repetition:02d}'
        age_predictions[repetition_column_name] = np.nan

        # Create 10-fold CV scheme stratified by age
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i_repetition)
        for i_fold, (train_index, test_index) in enumerate(skf.split(regions_norm, age)):
            print(f'Running repetition {i_repetition:02d}, fold {i_fold:02d}')

            x_train, x_test = regions_norm[train_index], regions_norm[test_index]
            y_train, y_test = age[train_index], age[test_index]

            # Scaling using inter-quartile range
            scaler = RobustScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            if model_name == 'LinearSVR':
                model_type = LinearSVR(loss='epsilon_insensitive', max_iter=100000)
                param_grid = {'C': [2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7]}
            elif model_name == 'SVR_rbf':
                model_type = SVR(kernel='rbf')
                param_grid = {'C': [2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 3, 2 ** 5]}
                # param_grid = {'C': [2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7]}
            elif model_name == 'XGBoost':
                # model_type = XGBRegressor(learning_rate=0.01, n_estimators=6000, max_depth=7, subsample)
                model_type = XGBRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [100, 300, 500, 1000],
                    # 'n_estimators': [200, 600, 1500],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    # 'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 9],
                    # 'max_depth': [5, 7, 9],
                    'subsample': [0.6, 0.8, 1.0],
                    # 'subsample': [0.9, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    # 'colsample_bytree': [0.8, 0.9, 1.0],
                    'min_child_weight': [1, 3, 5]
                }
            else:
                print('Error: Model name not recognised.')

            # Systematic search for best hyperparameters
            nested_skf = StratifiedKFold(n_splits=n_nested_folds, shuffle=True, random_state=i_repetition)
            gridsearch = GridSearchCV(model_type,
                                      param_grid=param_grid,
                                      scoring='neg_mean_absolute_error',
                                      refit=True, cv=nested_skf,
                                      verbose=3, n_jobs=-1)

            gridsearch.fit(x_train, y_train)

            print(gridsearch.best_params_)

            model = gridsearch.best_estimator_

            params_results = {'means': gridsearch.cv_results_['mean_test_score'],
                              'params': gridsearch.cv_results_['params']}

            predictions = model.predict(x_test)

            mean_brainage_gap = (predictions - y_test).mean()
            mae = mean_absolute_error(y_test, predictions)
            rmse = sqrt(mean_squared_error(y_test, predictions))
            r, _ = stats.pearsonr(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            age_error_corr, _ = stats.spearmanr((predictions - y_test), y_test)

            cv_mean_brainage_gap.append(mean_brainage_gap)
            cv_r.append(r)
            cv_r2.append(r2)
            cv_mae.append(mae)
            cv_rmse.append(rmse)
            cv_age_error_corr.append(age_error_corr)

            # Save model coefficients for this fold if using LinearSVR
            if model_name == 'LinearSVR':
                coefficients_list.append(model.coef_)

            # ----------------------------------------------------------------------------------------
            # Save output files
            output_prefix = f'{i_repetition:02d}_{i_fold:02d}'

            # Save scaler and model
            dump(scaler, cv_dir / f'{output_prefix}_scaler.joblib')
            dump(model, cv_dir / f'{output_prefix}_regressor.joblib')
            dump(params_results, cv_dir / f'{output_prefix}_params.joblib')

            # Save model scores
            scores_array = np.array([r, r2, mae, rmse, age_error_corr])
            np.save(cv_dir / f'{output_prefix}_scores.npy', scores_array)

            # ----------------------------------------------------------------------------------------
            # Add predictions per test_index to age_predictions
            for row, value in zip(test_index, predictions):
                age_predictions.iloc[row, age_predictions.columns.get_loc(repetition_column_name)] = value

            if model_name == 'LinearSVR':
                # Calculate Shapley values
                explainer = shap.Explainer(model, x_train)
                shap_values = explainer(x_test)
                shap_values_df.iloc[test_index] += shap_values.values

            # Print results of the CV fold
            print(f'Repetition {i_repetition:02d} Fold {i_fold:02d} '
                  f'Mean brain age gap: {mean_brainage_gap:0.3f}, '
                  f'r: {r:0.3f}, R2: {r2:0.3f}, '
                  f'MAE: {mae:0.3f} RMSE: {rmse:0.3f} CORR: {age_error_corr:0.3f}')

    # Save predictions
    age_predictions.to_csv(model_dir / 'age_predictions_train.csv')

    # --------------------------------------------------------------------------------------------
    print('************** CALCULATING MODEL PERFORMANCE **************')
    # Variables for mean scores of performance metrics of CV folds across all repetitions

    # Initialize an empty DataFrame to hold the performance results
    results = pd.DataFrame(columns=['Measure', 'Value'])

    # Define a list of dictionaries, each containing the measure and value
    # Please note: The std values calculated across folds below represents variability in model performance within those folds;
    # However, in the chapter, the overall std across the entire dataset is used, as calculated in  create_figures.py.
    rows = [
        {'Measure': 'mean_brainage_gap', 'Value': np.mean(cv_mean_brainage_gap)},
        {'Measure': 'mean_r', 'Value': np.mean(cv_r)},
        {'Measure': 'std_r', 'Value': np.std(cv_r)},
        {'Measure': 'mean_r2', 'Value': np.mean(cv_r2)},
        {'Measure': 'std_r2', 'Value': np.std(cv_r2)},
        {'Measure': 'mean_mae', 'Value': np.mean(cv_mae)},
        {'Measure': 'std_mae', 'Value': np.std(cv_mae)},
        {'Measure': 'mean_rmse', 'Value': np.mean(cv_rmse)},
        {'Measure': 'std_rmse', 'Value': np.std(cv_rmse)},
        {'Measure': 'mean_age_error_corr', 'Value': np.mean(cv_age_error_corr)},
        {'Measure': 'std_age_error_corr', 'Value': np.std(cv_age_error_corr)}
    ]

    # Convert the list of dictionaries to a DataFrame and concatenate with results
    results = pd.concat([results, pd.DataFrame(rows)], ignore_index=True)

    # Save results to CSV
    results.to_csv(model_dir / 'performance_scores_summary.csv', index=False)

    print(results)

    # ----------------------------------------------------------------
    # Optional: Calculate model coefficients for LinearSVR
    # These may be compared to the SHAP output
    if model_name == 'LinearSVR':
        print('************** CALCULATING MODEL COEFFICIENTS **************')
        # Calculate mean and std of coefficients across iterations
        coefficients_array = np.array(coefficients_list)
        mean_coefficients = coefficients_array.mean(axis=0)
        std_coefficients = coefficients_array.std(axis=0)

        # Save mean and std coefficients to a DataFrame
        coefficients_summary = pd.DataFrame({
            'Feature': COLUMNS_NAME,
            'Mean Coefficient': mean_coefficients,
            'Std Coefficient': std_coefficients,
            'Absolute Mean Coefficient': np.abs(mean_coefficients)
        })

        # Sort and save coefficients summary to CSV
        coefficients_summary = coefficients_summary.sort_values(by='Absolute Mean Coefficient', ascending=False)
        coefficients_summary.to_csv(model_dir / 'model_coefficients_summary.csv', index=False)

        # Plot top 10 features
        # Select the top 10 features with the highest absolute mean coefficient
        top_10_features = coefficients_summary.head(10).set_index('Feature')['Absolute Mean Coefficient']

        # Plotting the top 10 features
        plt.figure(figsize=(12, 6))
        top_10_features.plot(kind='bar', color='skyblue')
        plt.title('Top 10 Features Contributing to the Model (Based on Absolute Mean Model Coefficient)')
        plt.xlabel('Features')
        plt.ylabel('Average Absolute Mean Coefficient')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(model_dir / 'feature_importance_top10_absolute.png', format='png')
        plt.show()

    # ----------------------------------------------------------------
    # Shapley calculations
    # This is only done for LinearSVR, because it takes very long/does not work for any other model
    if model_name == 'LinearSVR':
        print('************** CALCULATING SHAP VALUES **************')

        # Average Shapley values across all models
        shap_values_df /= (n_folds * n_repetitions)

        # Calculate the average absolute Shapley values for each region
        average_shap_values = shap_values_df.abs().mean()

        # Get the top 10 regions with the highest average absolute Shapley values
        top_10_regions = average_shap_values.nlargest(10)

        # # Average SHAP values across repetitions
        # shap_values_avg = shap_values_df / (n_folds * n_repetitions)
        #
        # # Calculate the mean SHAP values for each feature
        # mean_shap_values = shap_values_avg.mean(axis=0)
        #
        # # Get the top 10 features with the highest mean absolute SHAP values but keep the original signs
        # top_10_features = mean_shap_values.abs().nlargest(10).index
        # sorted_mean_shap_values = mean_shap_values.loc[top_10_features]

        # Plotting the top 10 regions
        plt.figure(figsize=(12, 6))
        top_10_regions.plot(kind='bar')
        # sorted_mean_shap_values.plot(kind='bar')
        plt.title('Top 10 Regions Contributing to the Model (Based on SHAP)')
        plt.xlabel('Brain Regions')
        plt.ylabel('Average Absolute Shapley Value')
        # plt.ylabel('Mean SHAP value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        # plt.savefig(model_dir / 'shapley_training_positive_negative.png', format='png')
        plt.savefig(model_dir / 'shapley_training_absolute.png', format='png')


if __name__ == '__main__':
    train_model()
