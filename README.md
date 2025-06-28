# Wind Power Forecasting

**The Erdős Institute Data Science Bootcamp Summer 2025**

**Team Members:**
- [Desmond Coles](https://github.com/desmondcoles1)
- [Andrea Fodor](https://github.com/AndreaFodor)
- [Kavindra Nissanka](https://github.com/kvnissanka87)
- [Manimugdha Saikia](https://github.com/m-saikia)
- [Jaxon Shumaker](https://github.com/shumakerJ)

## Introduction

Wind power is the second-largest source of renewable energy for HydroQuébec, a public utility corporation that provides power to Canadians in Québec and exports to Northeast American Power Traders. HydroQuébec has contracts stating that they must buy power generated from the wind farms they contract with. Over time, wind power can vary significantly, but this is due to natural processes, and wind power generation cannot be adjusted in real-time to meet market demands, unlike some forms of non-renewable energy. Predicting the amount of power produced by wind farms in Quebec is a crucial metric for the company. Power generation forecasts allow HydroQuébec to allocate resources effectively and plan its economic dispatch accordingly.

On the other hand, power generation forecasts are also valuable to the wind farms and Northeast American power traders. Power grid storage is generally limited, and supply and demand for power must be carefully balanced, as wind power cannot be controlled; therefore, other power sources must be adjusted to prevent waste. Thus, accurate day-ahead predictions are necessary to help maintain a balanced energy budget for both energy providers and consumers. Wind power generation can be very low on some days, providing opportunities for wind farms to perform maintenance when it is safe to attend to individual wind turbines.

**Stakeholders:** Our stakeholders include HydroQuébec, Northeast American power traders, and companies that own wind farms in Québec.

## Goal
Use machine learning to predict one day in advance the total amount of wind power generated, given weather data in Québec.

**Key performance indicators:** Our Key performance indicators include the mean absolute percentage error, the mean absolute error on one-day-ahead power generation, and the R² Score. The MAPE provides an interpretable measure of average error. At the same time, the MAE allows someone with domain knowledge of the data to make informed judgments about accuracy, avoiding the issues the MAPE has with handling small values in the dataset. The R²-score helps us assess how well our model reflects the variance in the data, which is essential given the significant fluctuations in wind power.


## Data Sources
**Data for wind power production from HQ:** The dataset linked below provides the total amount of wind power generated for HydroQuébec each hour, from 2019 to 2023. This data is downloadable as a CSV file. 

**Source:** [https://donnees.hydroquebec.com/explore/dataset/historique-production-electricite-quebec/table/ ](https://donnees.hydroquebec.com/explore/dataset/historique-production-electricite-quebec/table/)


**Information about the wind farms:** The HydroQuébec website provides a list of all the wind farms that provide power to HydroQuébec. This information was scraped from the website to give us coordinates for the wind farms. There are 39 farms in total.

**Source:** [https://www.hydroquebec.com/electricity-purchases-quebec/electricity-contracts.html](https://www.hydroquebec.com/electricity-purchases-quebec/electricity-contracts.html)


**Weather data:** This data was obtained from Open-Meteo. This open-source meteorological tool allows anyone to retrieve a set of weather variables at any latitude and longitude coordinate, and for any year since 1940. Open-Meteo provides an easy-to-use API, which we interfaced with to automatically download temperature, relative humidity, wind speed at 10m, and wind direction data for each wind farm every hour, for the years 2019-2023. 

**Source:** [https://open-meteo.com/](https://open-meteo.com/)

**Wind Power Data for real-time forecasting:** The HydroQuébec website provides the real-time data for power generation; however, it only provides the most recent 2 days' worth of data. To run a real-time forecast, our models need at least a few weeks' worth of data to train on. The following website has an archive of the past year's data on wind-power generation, along with real-time generation data. We used this data for the version of the kNN regressor model that contains the code to make an API call to download the data, which subsequently goes through the pipeline to be preprocessed before being used to train the model for the real-time forecasting Python script. The website provides a free API key.

**Source:** [https://electricite-quebec.info/en](https://electricite-quebec.info/en) 


## Modeling Apprach
We retrieved temperature, relative humidity, wind speed, and wind direction data for each of the 39 HydroQuébec wind farm locations at one-hour intervals from 2019 to 2023. Wind direction was converted from an angle to two separate features that represented the x and y coordinates of a unit vector so that they could be scaled properly. We then combined this data into a CSV file that also contains the wind power generated by HydroQuébec for each hour. We set aside 2023 as a testing year for later reference.

Wind turbines generate power (P) by turning the kinetic energy (KE) of the air being pushed through the rotor blade area into electricity. Kinetic energy is proportional to $mv^2$, and the mass flow rate of air through a turbine is $\rho vA$. Thus, $P \propto \rho v^3A$. We don’t have the air density, but we know that the density of air is inversely proportional to temperature (assuming air is an ideal gas and follows PV=nRT), so we can create a new variable for each wind farm, $v^3A/T$. The rotor diameter would account for the area in this case. However, due to the non-uniformity of rotor diameter across the same farm and the fact that the kNN model did not seem to improve with the inclusion of the mean diameter data significantly, we decided to use the variable $v^3/T$ for each farm. We utilized this data to develop a new feature for each of the 39 locations.

From our exploratory data analysis steps, we realised that wind speed alone was one of the most essential features. Furthermore, Principal Component Analysis revealed that the explained variance when using 39 components was 98.8%, allowing us to reduce our feature space significantly. Two-dimensional distributions of the wind power with respect to some of our wind variables also suggest that wind speed is the major determining factor of wind power generation.


## Model Selection
We selected our models using the following cross-validation method: we trained them on data from 2019 to 2021 and then used them to predict values for the first six months of 2022. We then generated graphs of the predictions and calculated the MAPE, MAE, and the R²-score.

**Dummy model/baseline:** Our baseline model took the average value of all the training data as its future prediction. The results for MAPE, MAE, and the R²-score were 0.5702, 12660, and -0.006, respectively.

**Linear regression:** Basic linear regression with the mean daily wind speed across all farms as the regression variable. Resulted in MAPE of 0.1825, MAE of 4942, and R2 of 0.821.


**Basic LSTM Neural Network:** The dataset is treated as a time series and fed to a basic LSTM Neural Network. The best MAE obtained so far is ~16000, and as such, it does not perform better than a mean baseline. Therefore, we did not conduct any thorough analysis of the LSTM model.

**Multilinear Regression:** We have a basic MLR model that utilizes both the calendar date and historical weather data. This model attempts to predict wind power on the target day using only data and weather data from days preceding the target. The module did not perform well. 

**Random Forest:** For both the random forest regression and XGBoost, weather data from a single day was combined into a single data point for that day, and then PCA was applied to reduce the number of features to as few as six dimensions. This processing was designed to prevent decision trees from using features with low predictive power and to utilize all the information from a single day to make more accurate predictions for the next day. The MAPE, MAE, and the R²-score for our cross-validation scheme were 0.2485, 6034.039, and 0.7597, respectively.

**XGBoost:** The MAPE, MAE, and R² score for our cross-validation scheme were 0.2048, 5,284.8198, and 0.808, respectively.

**kNN Regressor:**  The kNN model approach treats the dataset as non-time series dependent (focusing on weather variables). The kNN model is trained on the engineered feature: (wind speed)$^3$ divided by temperature (in Kelvin) for each wind farm. We train on hourly data, predict on hourly data, and then add the predicted values for the 24 hours to get the prediction on a particular day. Each prediction is based on a rolling 60-day training window. Our cross-validation scheme's MAPE, MAE, and the R²-score were 0.153, 4037.2, and 0.887, respectively.

<center> 
<img src = "4_figures/knn_validation.png", width = '60%', height = '60%' />
</center>

**ARIMAx Time Series:** We worked with this model because it was autoregressive, and our data is from a time-dependent process. A regular ARIMA model was tested with various parameters and training windows. These models performed worse than the baseline model. In the end, ARIMAx model was selected with the mean daily wind speed across all wind farms added as an exogenous variable. The training window is 60 days prior to the prediction date. The model uses StatsForecast python library. The ARIMA model with the mean daily wind speed added as an exogenous variable. Training is done on 60 days prior to the prediction date. The (p, d, q) parameters of the model are optimized by autoARIMA for each prediction. The MAPE, MAE, and R² score for our cross-validation scheme were 0.1832, 4,570.28, and 0.847, respectively.

<center> 
<img src = "4_figures/arima_validation.png", width = '60%', height = '60%' />
</center>



## Test Results
We found that on the validation data, the models that performed best were ARIMAx and kNN. We wanted to use both a time series model and a non-time series model, since generated wind power may not necessarily depend on previous data points, but rather on the weather conditions of that day. Of course, the weather itself is dependent on the weather from the last day, so for that reason, we tried both.
 
These models were tested using the data from 2023. For ARIMAx, we found that the MAPE, MAE, and R²-score were 0.289, 4919, and 0.827, respectively. For kNN, we discovered that the MAPE, MAE, and R² Score were 0.218, 4,297, and 0.873, respectively. These results are slightly worse than those on our testing data, but still provide reasonably accurate results. The MAE for both models is less than half the average power generated per day, which is around 31000 MW. We scale our MAE by the maximum power generated across Québec, which is approximately 79,000 MW. In that case, we obtain a scaled MAE of less than 8%, which is better than the scaled MAE of the results for wind power forecasting in other regions, as reported in [this study](https://www.sciencedirect.com/science/article/pii/S0960148122004931). Though this MAE is still large enough that we may not always accurately predict very low power generation days. Our R² score is lower than [that of some other experiments](https://www.nature.com/articles/s41598-025-96262-6#:~:text=The%20accuracy%20and%20stability%20are%20better%20than,pivotal%20direction%20in%20wind%20power%20prediction%20research.), which have achieved R² scores as high as 0.98 for wind power forecasts. 

We choose our best model on the basis of MAPE, MAE and R²-scores from the validation step. We kept two models - one that treats the data as time-series and the other which does not. 
- **kNN Regressor (Time-series agnostic) Plots:** 

<center> 
<img src = "4_figures/knn_testing.png", width = '60%', height = '60%' />
</center>

- **ARIMAx (Time-series observant) Plots:** 

<center> 
<img src = "4_figures/arima_testing.png", width = '60%', height = '60%' />
</center>



## Dash APP
An interactive *dash app* that allows the user to choose a model between **kNN Regressor** (with two options: i\) "Validation/Testing" running `kNN_script`, and ii\) "Real-time forecasting" running `kNN_real_time_script`) and **ARIMAx**, and input a prediction window. The difference between "kNN (Validation/Testing)" and "kNN (Real-time Forecasting)" is that the latter, in theory, has all the necessary code to make a real-time forecast, given the user has access to the power data API key. 

<center>
<p> 
<img src = "4_figures/dash1.png", width = '40%' />
<img src = "4_figures/dash2.png", width = '40%' />
</p>
</center>


## Repository Structure

This repository contains the following:

- `1_data/`: Contains all the notebooks to download, clean and merge the final dataframe for both validation and testing. Includes a *README* explaining the structure of the folder.

- `2_modeling/:` Contains all the notebooks and scripts for exploratory data analysis, model selection and execution. Includes a *README* explaining the structure of the folder.

- `3_checkpoints/:` Various checkpoints throughout the project window.

- `4_figures/:` Useful figures about the dash app, and plots from the validation and testing of our best models.

- `enviroment.yml:` Exported `conda` environment file containing the dependencies.

