## Folder Structure

This folder contains implementation of different modeling approach that we tried. Our best models were **kNN Regressor** and **ARIMAx**.  

- `arima_model/`: Contains exploratory data analysis, initial trials of some time series models, and the main ARIMAx model.
    - `exploreData.ipynb`: contains exploratory data analysis of the dataset. 
    - `time_series_mode.ipynb`: contains initial trials of different time series model approaches.
    - `models/`: Contains Python script for the ARIMAx model and a few baseline models:
        - `baselines.py`: Baseline models.
        - `timeseries.py`: **ARIMAx** model.
- `knn_model/`: Contains all relevant folders for the **kNN Regressor** model - executable scripts, explanatory jupyter notebooks, and a *README* file for the folder.
- `linear_regression/`: Contains a jupyter notebook `LinearRegression.ipynb` highlighting the implementation of a simple linear regression model with only the mean wind speed across all wind farms as the feature.
- `multilinear_regression/`:
- `random_forests/`:
- `dash_app.py`: An interactive *dash app* that allows the user to choose a model between **kNN Regressor** (with two options: i\) "Validation/Testing" running `kNN_script`, and ii\) "Real-time forecasting" running `kNN_real_time_script`) and **ARIMAx**, and input a prediction window. The difference between "kNN (Validation/Testing)" and "kNN (Real-time Forecasting)" is that the former, in theory, can make real-time forecast, given the user has access to the power data API key. The app runs the chosen model on the prediction window to display the error scores and a plot of the predicted values vs the true values.
- `final_testing.ipynb`: A jupyter notebook displaying the error scores and plots by running the best models we have - **kNN Regressor** and **ARIMAx**.
- `power_output_trend.ipynb`: A jupyter notebook with some initial basic attempts at exploring the data.