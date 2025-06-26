## Folder Structure

This folder includes two folders showing two different implemetation based on two aspects:  
- `hyperparams_tuning_and_model_selection/`: Cross-validation,hyperparameter tuning, and feature selection based on available data between 2019 to 2023.
- `real_time_forecast/`: Real-time forecasting using the best model from above.

#### Structure of `hyperparams_tuning_and_model_selection/`:
- `kNN_script.py`: An executable Python script with the class `kNN_Cross_Validation`.
- `kNN_cross_validation.ipynb`: A jupyter notebook with explanation of the modeling approach, **documentation** of `kNN_script.py`, and example executions of available methods.

#### Structure of `real_time_forecast/`:
- `kNN_real_time_script.py`: An executable Python script with the class `kNN_forecast`. The script (**in theory**) downloads the relevant wind power output and weather data to train the model, along with weather forecast data for the prediction window. The **main difference** of this script with `kNN_script.py` is the ability of this script to produce forecast using real-time weather forecasting data, while the other one is used to study our best model.
- `kNN_real_time.ipynb`: A jupyter notebook **documentation** of `kNN_real_time_script.py`, and example executions of the available methods.
- `real_time_weather_api.py`: Contains the function `weather_api_call` to retrieve real time weather forecasting data using an API call.
- `dash_app_real_time.py`: An interactive *dash app* executable script to implement `kNN_real_time_script.py`, and to display a plot with forecasted values.
- `already_downloaded_data/:` In case, the user inputs a forecast window in the past, the script uses the already downloaded data, rather than an unnecessary API call. Contains two `.csv` files corresponding to weather and power data.

