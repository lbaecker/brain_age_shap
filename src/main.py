"""Main script to run the entire pipeline.

Steps:
(1) fetch_data
(2) resample_data (i.e., undersample Biobank Scanner01)
(3) run_combat (harmonisation for scanner and gender)
(4) train_model (train brain age prediction model in healthy subjects)
(5) apply_model (apply brain age prediciton model to clinical datasets incl. patients and healthy subjects)
(6) run_shap_analyis (calculate regional contributions to model)

Please note: The datasets referred to as validation in this repo are called clinical data in the chapter.
"""
from src.fetch_data import create_data_file
from src.resample_data import resample_data
from src.run_combat import run_combat
from src.test_harmonization import test_harmonization
from src.train_model import train_model
from src.apply_model import apply_model
from src.statistical_analysis import analyse_brainage
from src.shapley_analysis import run_shap_analysis


def run_brain_age():
    # Define directory where all data are saved
    data_dir = '/home/kcl/git/fresh_validation/validation_study_2020/data/raw/'

    # Define datasets to include in this project (training and validation)
    dataset_list = ['ABIDEII', 'ADHD200', 'ADNI2', 'ADNI3', 'ADNIGO', 'AIBL', 'AssociativeLearning',
                    'BIOBANK', 'COBRE', 'Dublin', 'EUGEI', 'FalseBeliefs',
                    'Galway', 'Gottingen', 'HarmAvoidance', 'HCP', 'HCP-PSYCHOSIS', 'HMRRC', 'HumanConnectomeProject-Aging',
                    'IOPPN', 'IXI', 'LossAdversion',
                    'MaastrichtGROUP', 'MaastrichtUniversity', 'MaturationalChanges', 'MCIC', 'MoralJudgment',
                    'NUSDAST', 'PAFIP', 'RouteLearning', 'SequentialInferenceVBM',
                    'TOMC', 'UCLA', 'UCLDevlin', 'UtrechtGROUP', 'WashingtonUniversity']

    # Define dataset names to be used for validation only, i.e., clinical datasets incl. patients and healthy controls
    validation_list = ['ABIDEII', 'COBRE',
                       'EUGEI', 'Galway', 'Gottingen', 'HCP-PSYCHOSIS', 'HMRRC',
                       'IOPPN', 'MCIC', 'TOMC']

    # Dataset preparation
    data_path = create_data_file(data_dir, dataset_list, validation_list,
                                 mriqc_threshold=0.7, min_age=8, max_age=85,
                                 min_n_per_scanner_train=20, min_n_per_scanner_val=1)
    data_resampled_path = resample_data(data_path, n_biobank=100)
    data_harmonised_path = run_combat(data_resampled_path)

    # Run K-S test for harmonisation
    test_harmonization(data_harmonised_path)

    # Define variables for model development and analysis
    # model_name = 'XGBoost'
    model_name = 'LinearSVR'
    # model_name = 'SVR_rbf'
    n_repetitions = 10
    # n_folds = 10
    n_folds = 10

    # Specify whether Shapley analysis should be conducted
    calculate_shap = True
    # calculate_shap = False

    # Developing the brain age model on healthy control data
    train_model(data_harmonised_path, model_name=model_name, n_repetitions=n_repetitions, n_folds=n_folds)

    # Apply the trained model to clinical datasets including healthy controls and patients
    agepredictions_val_path = apply_model(data_harmonised_path, model_name=model_name, n_repetitions=n_repetitions,
                                          n_folds=n_folds, calculate_shap=calculate_shap)

    # Linear regression for patients vs controls, controlling for age and gender
    analyse_brainage(agepredictions_val_path, model_name=model_name)

    # Analyse differences between diagnostic groups using Shapley, if calculate_shap=True
    if calculate_shap is True:
        run_shap_analysis(data_harmonised_path, model_name=model_name)


if __name__ == '__main__':
    run_brain_age()
