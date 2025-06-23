import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from datetime import date
from real_time_weather_api import weather_api_call
from hydroquebec.api import Hydro_quebec_data
from typing import Tuple, List, Union, Dict

## Helper Functions

## Provides the list of days in yy-mm-dd format within the time_window
def list_of_days(time_window: pd.DatetimeIndex) -> List[str]:
    assert type(time_window) ==  pd.core.indexes.datetimes.DatetimeIndex, ("TypeError: "
                                "time_window must be of type pandas.core.indexes.datetimes.DatetimeIndex.")
    return [x.strftime('%y-%m-%d') for x in time_window]

## For feature engineering, speed cubed divided by the temperature in Kelvin is proved to be beneficial
## and also supported by the theoretical formula
def speed_cubed_div_temp(df: pd.DataFrame) -> pd.DataFrame:    
    new_cols = []
    for i in range(1, 40):
        spd_col = f'wind_speed_10m_{i}'
        temp_col = f'temperature_2m_{i}'
        final_col = (df[spd_col] ** 3)/ (df[temp_col].values + 273) #divide by temperature in Kelvin scale   
        new_cols.append(pd.DataFrame({
            f'{spd_col}_cubed_div_temp': final_col
        }))

    df_new = pd.concat([df.drop(columns= [f'wind_speed_10m_{i}' for i in range(1, 40)] +
                                [f'temperature_2m_{i}' for i in range(1,40)] +
                                [f'wind_direction_10m_{i}' for i in range(1,40)]+
                                [f'relative_humidity_2m_{i}' for i in range(1,40)], errors='ignore')] + new_cols, axis=1)
    return df_new.copy()

## Makes a plot of the true values (trained_on) in the train_window
## and then extend (with a different legend) the predicted values
def plot(preds: List[float], trained_on: List[float], train_window: pd.DatetimeIndex, 
         predict_window: pd.DatetimeIndex) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    train_days = [x.strftime('%Y-%m-%d') for x in train_window] 
    predict_days = [x.strftime('%Y-%m-%d') for x in predict_window]
    days = train_days + predict_days


    ax.plot(train_days, trained_on, label = 'Trained data')
    ax.plot(predict_days, preds, label = "Predicted")
    ax.legend(loc='lower right')
    ax.tick_params(axis='x', rotation = 45)
    ax.set_xticks([days[i] for i in range(1,len(days),len(days)// 12)])
    ax.set_xlabel('Date')
    ax.set_ylabel('Total power per day (in MW)')
    ax.set_title("Plot of total power per day showing the trained data and the predicted data.")

    plt.show()



def read_and_clean_power_data(train_window: pd.DatetimeIndex) -> pd.DataFrame:
    power_df = pd.read_csv("real_time_data/real_time_power_data.csv")[["time","wind"]]

    power_df['time'] = pd.to_datetime(power_df['time'])
    power_df['time'] = power_df['time'].apply(lambda x: x.floor('h'))

    power_df = power_df[power_df['time'].isin(pd.date_range(train_window[0], 
                                            train_window[-1].replace(hour=23), freq='h'))]
    power_df = power_df.sort_values('time')

    #Checking duplicate 'time' in the power dataset
    if len(power_df[power_df['time'].duplicated()]) != 0:
        print("Duplicates detected in power data. Calling duplicate handling funcion...")
        power_df = power_duplicate_handle(power_df)

    data_already_downloaded_till = pd.Timestamp('2025-06-20')     ## the date till which we already have the date available, 
                                                    ## setting '2025-06-20' for now
    if train_window[-1] <= data_already_downloaded_till: 
                                    ## if the predict window is before 'data_already_downloaded_till'
                                    ## we'll use the already downloaded data to train and predict.
                                    ## This will improve the runtime significantly as most of the data
                                    ## will already be downloaded apart from a few days
                                                        
        return power_df
    else:
        start = data_already_downloaded_till + pd.Timedelta(1, unit='d')
        end = train_window[-1].strftime('%Y-%m-%d')            
        new_rows =  power_api_call(start,end)       
        new_power_df = pd.merge(power_df, new_rows, on=['time'])
        if len(new_power_df[new_power_df['time'].duplicated()]) != 0:
            print("Duplicates detected in downloaded power data. Calling duplicate handling funcion...")
            new_power_df = power_duplicate_handle(new_power_df)
        return new_power_df

## in case, there are any duplicate time entries    
def power_duplicate_handle(df: pd.DataFrame, time_window: pd.DatetimeIndex) -> pd.DataFrame:
    ## body of the code here
    return df

## start_date and end_date in 'YYYY-MM-DD' format
def power_api_call(start_date, end_date): 
    api_key = 'API_key_here'
    data_type = 'generation'
    data_frame = Hydro_quebec_data(api_key, data_type, start_date, end_date)
    return data_frame ### may need to convert to pd.Dataframe as the api call may return a dictionary



def read_and_clean_weather_data(train_window: pd.DatetimeIndex, predict_window: pd.DatetimeIndex) -> pd.DataFrame:
    weather_df = pd.read_csv("real_time_data/real_time_weather_data.csv")

    train_window = list_of_days(train_window)
    predict_window = list_of_days(predict_window)
    time_window = pd.to_datetime(train_window + predict_window)
    
    weather_df['time'] = pd.to_datetime(weather_df['time'])
    weather_df['time'] = weather_df['time'].apply(lambda x: x.floor('h'))

    weather_df = weather_df[weather_df['time'].isin(pd.date_range(time_window[0], 
                                            time_window[-1].replace(hour=23), freq='h'))]
    weather_df = weather_df.sort_values('time')

    #Checking duplicate 'time' in the power dataset
    if len(weather_df[weather_df['time'].duplicated()]) != 0:
        print("Duplicates detected in weather data. Calling duplicate handling funcion...")
        weather_df = weather_duplicate_handle(weather_df, time_window)

    data_already_downloaded_till = pd.Timestamp('2025-06-20') 
    if time_window[-1] <= data_already_downloaded_till: ## if the predict window is before '2025-06-20'
                                                        ## we'll use the already downloaded data to train and predict.
                                                        ## This will improve the runtime significantly as most of the data
                                                        ## will already be downloaded apart from a few days 
        return weather_df
    else:           ## if some rows don't exist, download them, merge and return the new dataframe
        start = data_already_downloaded_till + pd.Timedelta(1,unit='d') 
        end = time_window[-1].strftime('%Y-%m-%d')
        new_rows = weather_api_call(start, end)     
        new_weather_df = pd.merge(weather_df, new_rows, on= ['time'])
        if len(new_weather_df[new_weather_df['time'].duplicated()]) != 0:
            print("Duplicates detected in downloaded weather data. Calling duplicate handling funcion...")
            new_weather_df = weather_duplicate_handle(new_weather_df, time_window)
        return new_weather_df
    
def weather_duplicate_handle(df: pd.DataFrame, time_window: pd.DatetimeIndex) -> pd.DataFrame:
    ## body of the code here
    return df

def merge_power_weather_for_appropriate_time_window() -> (
        Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]):
    predict_window = get_forecast_window()
    train_window = get_train_window(predict_window)

    power_df = read_and_clean_power_data(train_window)          ## already only has the training data
    weather_df = read_and_clean_weather_data(train_window, predict_window)
    
    weather_df_train = weather_df[weather_df['time'].isin(pd.date_range(train_window[0], 
                                            train_window[-1].replace(hour=23), freq='h'))]
    train_df = pd.merge(power_df, weather_df_train, on=['time'])

    predict_df= weather_df[weather_df['time'].isin(pd.date_range(predict_window[0], 
                                            predict_window[-1].replace(hour=23), freq='h'))]    
                ## assuming in real-time case, we won't have the wind power column in the predict_df
                ## that's the entire point that we are predicting anyway
    return train_window, predict_window, train_df, predict_df



def get_forecast_window() -> pd.DatetimeIndex:
    """
    Interactively prompts the user to input a forecast window.

    Returns:
        pd.DatetimeIndex: A pandas date range between selected start and end date.
    """
    earliest_allowed_date = pd.Timestamp('2019-03-02')
    latest_allowed_date = pd.Timestamp(f'{str(date.today())}')
    print(f"\nPlease enter forecast start date in the format YYYY-MM-DD." 
          f"The date should be in between {earliest_allowed_date.strftime('%Y-%m-%d')}"
          f" to {latest_allowed_date.strftime('%Y-%m-%d')}.")
    date_str= input("Input the start date now: ")
    try:
        year, month, day = int(date_str[:4]), int(date_str[5:7]), int(date_str[8:])
        start_date = pd.Timestamp(year=year, month=month, day=day)
        assert earliest_allowed_date <= start_date <= latest_allowed_date, ("ValueError: "
        f"Please input a date between {earliest_allowed_date.strftime('%Y-%m-%d')}"
        f" to {latest_allowed_date.strftime('%Y-%m-%d')}.")
    except ValueError:
        raise AssertionError(f"Invalid date: {year}-{month}-{day}")
    print(f"\nYou have entered: {start_date.strftime('%Y-%m-%d')}.")

    window_length = 3
    print(f"\nPlease enter forecast window (in days). It should be an integer between 1 or {window_length}.")
    
    window = int(input("Input forecast window: "))
    assert 1 <= window <= window_length, f"ValueError: Input should be an integer between 1 or {window_length}."

    end_date = start_date + pd.Timedelta(window-1, unit='d')
    predict_window = pd.date_range(start=start_date, end=end_date, freq='d')
    
    print(f"\nYour forecast window is days between: {predict_window[0].strftime('%Y-%m-%d')} "
          f"and {predict_window[-1].strftime('%Y-%m-%d')}\n")
    return predict_window

def get_train_window(predict_window: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Computes the time window to train on, which either starts from 2019-01-01 or starts from 60 days
    prior to predict window.

    Args:
        predict_window (pd.DatetimeIndex): The window to predict on.
    
    Returns:
        train_window (pd.DatetimeIndex): The training dataset window.
    """
    days_to_train_on = 60
    predict_start_date = predict_window[0]
    time_diff = predict_start_date - pd.Timestamp(year=2019,month=1,day=1)
    if time_diff.days <= days_to_train_on:
        train_start_date = pd.Timestamp(year=2019,month=1,day=1)
    else:
        train_start_date = predict_start_date - pd.Timedelta(days_to_train_on, unit='d')
    train_window = pd.date_range(start= train_start_date, 
                                 end= predict_start_date - pd.Timedelta(1, unit='d'), freq='d')
    return train_window       


## Takes in training and testing dataframes and a hyperparams combination of PCA and knn-neighbours
## and returns an array of predicted values corresponding to each day.
## Note: training is done on hourly data, predicted on hourly data and finally added the predicted values per day
def kNN_main(pca_comp = 35, n_nbr = 5 ):

    train_window, predict_window, train_df, predict_df = merge_power_weather_for_appropriate_time_window()
    train_df = speed_cubed_div_temp(train_df)
    predict_df = speed_cubed_div_temp(predict_df)

    features = train_df.columns.drop(['Unnamed: 0.1','Unnamed: 0', 'time', 'Year', 'YearMonthDay', 'Wind'], 
                                     errors='ignore')

    pipe = Pipeline([('scale', StandardScaler()), ('pca', PCA(n_components= pca_comp))])
    pipe.fit(train_df[features])

    pca_train = pipe.transform(train_df[features])
    pca_predict = pipe.transform(predict_df[features])

    knn = KNeighborsRegressor(n_neighbors=n_nbr)
    knn.fit(pca_train, train_df.Wind)
    pred = knn.predict(pca_predict)

    trained_on = np.array(train_df.Wind)
    trained_on = [sum(trained_on[i:i+24]) for i in range(0,len(trained_on),24)] ## aggregate of each day 

    predicted = [sum(pred[i:i+24]) for i in range(0, len(pred),24)] ## add all the hourly data to get the aggregate for each day
    plot(predicted, trained_on, train_window, predict_window)       ## make the plot
    return predicted