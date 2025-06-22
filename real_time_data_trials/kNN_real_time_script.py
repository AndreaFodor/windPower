import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


## Helper Functions

## Provides the list of days in yy-mm-dd format within the time_window
def list_of_days(time_window):
    assert type(time_window) ==  pd.core.indexes.datetimes.DatetimeIndex, ("TypeError: "
                                "time_window must be of type pandas.core.indexes.datetimes.DatetimeIndex.")
    return [x.strftime('%y-%m-%d') for x in time_window]

## For feature engineering, speed cubed divided by the temperature in Kelvin is proved to be beneficial
## and also supported by the theoretical formula
def speed_cubed_div_temp(df):    
    new_cols = []
    for i in range(1, 40):
        spd_col = f'wind_speed_10m_{i}'
        temp_col = f'temperature_2m_{i}'
        final_col = (df[spd_col] ** 3)/ (df[temp_col].values + 273) #divide by temperature in Kelvin scale   
        new_cols.append(pd.DataFrame({
            f'{spd_col}_cubed_div_temp': final_col
        }))

    # Concatenate all new columns at once
    df_new = pd.concat([df.drop(columns= [f'wind_speed_10m_{i}' for i in range(1, 40)] 
                                + [f'temperature_2m_{i}' for i in range(1, 40)] )] + new_cols, axis=1)
    final_df = df_new.copy()  # defragment
    return final_df




def get_forecast_window():
    print("Please enter forecast start date in the format YYYY-MM-DD. The date should be in between 2020-01-01 to 2023-12-31.")
    date= input("Input the start date now: ")
    try:
        year = int(date[:4])
        month = int(date[5:7])
        day = int(date[8:])
        assert 2020 <= year <=2023, "ValueError: Input a year between 2020 to 2023."
        start_date = pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        raise AssertionError(f"Invalid date: {year}-{month}-{day}")
    print(f"You have entered: {start_date.strftime('%y-%m-%d')}.")

    window_length = 365
    print(f"Please enter forecast window (in days). It should be an integer between 1 or {window_length}.")
    window = int(input("Input forecast window: "))
    assert 1 <= window <= window_length, f"ValueError: Input should be an integer between 1 or {window_length}."
    end_date = start_date + pd.Timedelta(window, unit='d')
    predict_window = pd.date_range(start=start_date, end=end_date, freq='d')
    print(f"Your forecast window is days between: {predict_window[0]} and {predict_window[-1]}\n\n")

    ## whatever the test_window is, the train_window would be all the days 2 years prior to the start_date of predict_window
    ## In case, 2 years is not possible since we only have data from 2019-01-01, we adjust that here
    time_diff = start_date - pd.Timestamp(year=2019,month=1,day=1)
    if time_diff.days <= 730:
        train_start_date = pd.Timestamp(year=2019,month=1,day=1)
    else:
        train_start_date = start_date - pd.Timedelta(730, unit='d')
    train_window = pd.date_range(start= train_start_date, 
                                 end= start_date - pd.Timedelta(1, unit='d'), freq='d')
    return train_window, predict_window        

def get_data():
    pca_comp = 35       ## default values (found to be the best hyperparameters)
    n_nbr = 5           

    train_df, predict_df = merge_power_weather_for_appropriate_time_window()
    train_df = speed_cubed_div_temp(train_df)
    predict_df = speed_cubed_div_temp(predict_df)
    
    return train_df, predict_df, pca_comp, n_nbr



## Makes a plot of the true values (trained_on) in the train_window
## and then extend (with a different legend) the predicted values
def plot(preds, train_df):
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    train_window, predict_window = get_forecast_window()

    train_days = list_of_days(train_window)
    predict_days = list_of_days(predict_window)

    trained_on = np.array(train_df.Wind)

    ax.plot(train_days, [sum(trained_on[i:i+24]) for i in range(0,len(trained_on),24)], 
            label = 'Trained data')
    ax.plot(predict_days, preds, label = "Predicted")
    ax.legend(loc='lower right')
    ax.tick_params(axis='x', rotation = 45)
    ax.set_xticks([])
    ax.set_xlabel('Date')
    ax.set_ylabel('Total power per day (in MW)')
    ax.set_title("Plot of total power per day showing the trained data and the predicted data.")

    plt.show()







## Reading, cleaning and merging weather and power data


def read_and_clean_power_data():
    _, predict_window = get_forecast_window()
    power_df = pd.read_csv("real_time_data/real_time_power_data.csv")[["time","wind"]]
    
    #Checking duplicate 'time' in the power dataset
    if len(power_df[power_df['time'].duplicated()]) != 0:
        print("Duplicates detected in power data. Calling duplicate handling funcion...")
        power_df = power_duplicate_handle(power_df)
    power_df['time'] = pd.to_datetime(power_df['time'])
    power_df['time'] = power_df['time'].apply(lambda x: x.floor('h'))
    power_df = power_df.sort_values('time')

    data_already_downloaded_till = '2025-06-20'     ## the date till which we already have the date available, 
                                                    ## setting '2025-06-20' for now
    if predict_window[0] <= pd.Timestamp(data_already_downloaded_till): 
                                    ## if the predict window is before 'data_already_downloaded_till'
                                    ## we'll use the already downloaded data to train and predict.
                                    ## This will improve the runtime significantly as most of the data
                                    ## will already be downloaded apart from a few days
                                                        
        return power_df
    else:
        start = data_already_downloaded_till
        end = predict_window[0] - pd.Timedelta(1, unit='d') ## need the power data just one day before the predict_window            
        end = end.strftime('%y-%m-%d')
        new_rows =  power_api_call(start,end)       
        new_power_df = pd.merge(power_df, new_rows, on=['time'])
        return new_power_df

## in case, there are any duplicate time entries    
def power_duplicate_handle(df):
    ## body of the code here
    return df

from hydroquebec.api import Hydro_quebec_data
## start_date and end_date in 'YYYY-MM-DD' format
def power_api_call(start_date, end_date): 
    api_key = 'API_key_here'
    data_type = 'generation'
    data_frame = Hydro_quebec_data(api_key, data_type, start_date, end_date)
    return data_frame


def read_and_clean_weather_data():
    _, predict_window = get_forecast_window()
    weather_df = pd.read_csv("real_time_data/real_time_weather_data.csv")
    #Checking duplicate 'time' in the power dataset
    if len(weather_df[weather_df['time'].duplicated()]) != 0:
        print("Duplicates detected in weather data. Calling duplicate handling function...")
        weather_df = weather_duplicate_handle(weather_df)
    weather_df['time'] = pd.to_datetime(weather_df['time'])
    weather_df['time'] = weather_df['time'].dt.tz_localize(None)
    weather_df = weather_df.sort_values('time')
    if predict_window[0] <= pd.Timestamp('2025-06-20'): ## if the predict window is before '2025-06-20'
                                                        ## we'll use the already downloaded data to train and predict.
                                                        ## This will improve the runtime significantly as most of the data
                                                        ## will already be downloaded apart from a few days 
        return weather_df
    else:
        start = predict_window[0].strftime('%y-%m-%d')
        end = predict_window[-1].strftime('%y-%m-%d')
        new_rows = weather_api_call(start, end)       ## weather_api_call.py would be another module that provides these new_rows
        new_weather_df = pd.merge(weather_df, new_rows, on= ['time'])

def weather_duplicate_handle(df):
    ## body of the code here
    return df




def merge_power_weather_for_appropriate_time_window():
    power_df = read_and_clean_power_data()
    weather_df = read_and_clean_weather_data()
    train_window, predict_window = get_forecast_window()
    power_df_train = power_df[power_df['time'].isin(train_window)]
    weather_df_train = weather_df[weather_df['time'].isin(train_window)]
    train_df = pd.merge(power_df_train, weather_df_train, on=['time'])

    predict_df= weather_df[weather_df['time'].isin(predict_window)]     ## assuming in real-time case, we won't have the wind power
                                                                        ## column in the predict_df
                                                                        ## that's the entire point that we are predicting anyway
    return train_df, predict_df









## The main function

## Takes in training and testing dataframes and a hyperparams combination of PCA and knn-neighbours
## and returns an array of predicted values corresponding to each day.
## Note: training is done on hourly data, predicted on hourly data and finally added the predicted values per day
def kNN_main():
    df_train, df_predict, pca_comp, n_nbr = get_data()
    features = df_train.columns.drop(['Unnamed: 0', 'time', 'Year', 'YearMonthDay', 'Wind']) #extract the features

    pipe = Pipeline([('scale', StandardScaler()), ('pca', PCA(n_components= pca_comp))])
    pipe.fit(df_train[features])

    pca_train = pipe.transform(df_train[features])
    pca_predict = pipe.transform(df_predict[features])

    knn = KNeighborsRegressor(n_neighbors=n_nbr)
    knn.fit(pca_train, df_train.Wind)
    pred = knn.predict(pca_predict)

    predicted = [sum(pred[i:i+24]) for i in range(0, len(pred),24)] ## add all the hourly data to get the aggregate for each day
    plot(predicted, df_train)       ## make the plot
    return predicted