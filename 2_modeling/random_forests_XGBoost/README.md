## Folder Structure

This folder contains work on Random Forests and XGBoost. Random forests is an 'ensemble' ML technique using decision trees, where a collection of decision trees are made from the data, and they 'vote' on a prediction. XGBoost is a gradient boosting technique which begins with a simple decision tree.
 
The folder contains the following:

- `EDA_and_small_models.ipynb` some descriptive statistics are done using the statsmodel package and a decision tree model and linear regression model are run on the data.

- `Dummy_model_aka_baseline.ipynb` a baseline model using the mean power generation is run on the training and test data.

- `Random_Forest_hyperparameter_selection_and_CV.ipynb` contain some experiments used for hyperparmeter selection and an implementation of time series cross validation

- `Random_Forests_and_XGBoost_final_cross_validation.ipynb` after hyper parameters were selected for XGBoost and Random Forest we ran our cross validation scheme on them with MAPE, MAE, and R2 score.










