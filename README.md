# Repository for clinical application of brain age model, including Shapley analysis for regional contributions to the model

Steps involved:
- Load FreeSurfer-preprocessed data
- Basic data cleaning (as previously done in Vieira et al. 2025, Translational Psychiatry)
- Apply ComBat harmonization
- Train brain age model (SVR, as done previously in Baecker et al. 2021, HBM)
- Apply to clinical datasets 
- Conduct statistical analysis of BAG (OLS regression)
- Conduct statistical analysis of SHAP


Order of scripts:
- fetch_data
- resample_data
- run_combat
- test_harmonization
- train_model
- apply_model
- statistical analysis
- shapley_analysis