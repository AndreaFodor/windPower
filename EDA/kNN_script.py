import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from seaborn import set_style
import seaborn as sns

set_style("whitegrid")

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_percentage_error, r2_score


## Helper Functions

## Provides the list of days in yy-mm-dd format within the time_window
def list_of_days(time_window):
    assert type(time_window) ==  pd.core.indexes.datetimes.DatetimeIndex, ("TypeError: "
                                "time_window must be of type pandas.core.indexes.datetimes.DatetimeIndex.")
    return [x.strftime('%y-%m-%d') for x in time_window]


## Makes a column YearMonthDay: adds year in the MonthDay column in the format YY-MM-DD (eg 22-01-01)
## renames the column as YearMonthDay
def make_yymmdd_format(df):
    try:
        new_df = df
        new_df.time = pd.to_datetime(new_df.time).dt.tz_localize(None)
        #new_df = new_df.set_index('time')
        new_df['YearMonthDay'] = new_df['time'].apply(lambda x: x.strftime('%y-%m-%d'))
        new_df = new_df.drop(columns = ['MonthDay'])
        return new_df
    except:
        print("Incorrect dataframe passed to make_yymmdd_format.")

## Finds the true aggregate values of wind production per day of a given year
def true_aggregate_per_day(df, time_window):
    try:
        assert type(time_window) ==  pd.core.indexes.datetimes.DatetimeIndex, ("TypeError: "
                                "time_window must be of type pandas.core.indexes.datetimes.DatetimeIndex.")
        day_list = list_of_days(time_window)
        df_test = df[df.YearMonthDay.isin(day_list)]
        aggregates = {}
        for day in day_list:
            df_day = df_test[df_test['YearMonthDay'] == day]
            aggregates[day] = sum(np.array(df_day.Wind))
        return(aggregates)
    except:
        print("Incorrect dataframe passed to true_aggregate_per_day.")


## Splits train-test set based on start_date and end_date.
## The days between start_date and end_date are in the test set
## The days less than start_date are in the training set
def extract_train_test_data(df, start_date: pd.Timestamp, end_date = None):
    try:
        if end_date == None:
            end_date = start_date
        assert (type(start_date) == pd.Timestamp and type(end_date) == pd.Timestamp), ("TypeError: "
                                            "start_date and  end_date must be pandas.Timestamp type variable.")
        train_df = df[df['time'] < start_date]
        end_date = end_date.replace(hour=23)        ## by default the hour is 00:00:00, so replaced to make it 23:00:00
        test_df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
        return train_df, test_df
    except:
        print("Incorrect dataframe passed to extract_train_test_data.")

## This function calculates the cube of speed and divide it by the temperature (in Kelvin),
## adds this column to the dataframe and
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
    df_new = pd.concat([df.drop(columns= [f'wind_speed_10m_{i}' for i in range(1, 40)])] + new_cols, axis=1)

    final_df = df_new.copy()  # defragment
    return final_df

## This function checks for TypeError and ValueError in the variable test_window
## Need to train the model on at least on year on data and we don't have power data after 2023, 
## so the test_window is set to satisfy between 2020-01-01 to 2023-12-31
def predict_window_validity(test_window):
    assert type(test_window) == pd.core.indexes.datetimes.DatetimeIndex, ("TypeError: test_window must be of type "
            "pandas.core.indexes.datetimes.DatetimeIndex")
    test_window = test_window.sort_values()
    assert (pd.Timestamp(year=2020, month=1, day=1) <= test_window[0] 
            and test_window[-1] <= pd.Timestamp(year=2023, month=12, day=31)), ("ValueError: test_window must be "
            "between 2020-01-01 to 2023-12-31")

def get_forecast_window():
    print("Please enter forecast start date in the format YYYY-MM-DD. The date should be in between 2020-01-01 to 2023-12-31.")
    date= input("Input the start date now: ")
    try:
        year = int(date[:4])
        month = int(date[5:7])
        day = int(date[8:])
        assert 2019 <= year <=2023, "ValueError: Input a year between 2019 to 2023."
        start_date = pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        raise AssertionError(f"Invalid date: {year}-{month}-{day}")
    print(f"You have entered: {start_date.strftime('%y-%m-%d')}.")

    time_diff = pd.Timestamp(year=2022,month=12,day=31) - start_date  ### need to change this to 2023 later
    window_length = min(364, time_diff.days)
    print(f"Please enter forecast window (in days). It should be an integer between 1 or {window_length}.")
    window = int(input("Input forecast window: "))
    assert 1 <= window <= window_length, f"ValueError: Input should be an integer between 1 or {window_length}."
    end_date = start_date + pd.Timedelta(window, unit='d')
    predict_window = pd.date_range(start=start_date, end=end_date, freq='d')
    print(f"Your forecast window is days between: {predict_window[0]} and {predict_window[-1]}\n\n")
    return predict_window        

def get_data():
    pca_comp = 35       ## default values (found to be the best hyperparameters)
    n_nbr = 5           
    file_path = os.path.join(os.path.dirname(__file__), "../data/final_dataframes/main_training_dataframe.csv")
                                    ## need to include the main dataframe instead of the training set
    df = speed_cubed_div_temp(make_yymmdd_format(pd.read_csv(file_path)))
    df = df.drop(columns= [f'temperature_2m_{i}' for i in range(1,40)] +
                        [f'wind_direction_10m_{i}' for i in range(1,40)]+
                        [f'relative_humidity_2m_{i}' for i in range(1,40)])
    
    predict_window = get_forecast_window()
    return df, predict_window, pca_comp, n_nbr

## For each day in the test_window, this function trains a knn regressor model on all the day before that day,
## predicts on that day. Returns an array of predicted values for each day and an array with true aggregate for that day
def knn_using_particular_hyperparams_and_test_window(df, pca_comp, n_nbr, test_window):
    preds = []
    true_aggregates = []

    for test_day in test_window:

        true_aggregate = [float(x) for x in true_aggregate_per_day(df,pd.date_range(start=test_day,end=test_day, freq='d')).values()] 
                            # for just a day this array would have just 1 entry
        true_aggregates.append(true_aggregate[0])

        df_train, df_test = extract_train_test_data(df, test_day)
        pred = kNN_on_particular_train_test_splits(df_train, df_test, pca_comp, n_nbr)
                            # for just a day this array would have just 1 entry
        preds.append(pred[0])

        #print(f"Hyperparameters: {(pca_comp,n_nbr)}, test day: {test_day.strftime('%y-%m-%d')}, y_pred = {pred[0]:.1f}, y_true = {true_aggregate[0]:.1f}.")

    return preds, true_aggregates


## Takes in training and testing dataframes and a hyperparams combination of PCA and knn-neighbours
## and returns an array of predicted values corresponding to each day.
## Note: training is done on hourly data, predicted on hourly data and finally added the predicted values per day
def kNN_on_particular_train_test_splits(df_train, df_test, pca_comp, n_nbr):
    features = df_train.columns.drop(['Unnamed: 0', 'time', 'Year', 'YearMonthDay', 'Wind']) #extract the features

    pipe = Pipeline([('scale', StandardScaler()), ('pca', PCA(n_components= pca_comp))])
    pipe.fit(df_train[features])

    pca_train = pipe.transform(df_train[features])
    pca_test = pipe.transform(df_test[features])


    knn = KNeighborsRegressor(n_neighbors=n_nbr)
    knn.fit(pca_train, df_train.Wind)
    pred = knn.predict(pca_test)

    pred_per_day = [sum(pred[i:i+24]) for i in range(0, len(pred),24)]
    return pred_per_day


## The main function
def main():
    df, predict_window, pca_comp, n_nbr = get_data()
    predict_window_validity(predict_window)       ## testing that test_window is valid

    preds, true_aggregates = knn_using_particular_hyperparams_and_test_window(df, pca_comp, n_nbr, predict_window)

    mape= mean_absolute_percentage_error(y_pred = preds, y_true = true_aggregates)
    r2 = r2_score(y_pred = preds, y_true = true_aggregates)

    print(f"Mape = {mape:.3f}, R2 score = {r2:.3f}.")


    return  preds, true_aggregates, mape, r2