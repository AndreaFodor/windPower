import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from datetime import date
from real_time_weather_api import weather_api_call
from hydroquebec.api import Hydro_quebec_data
from typing import Tuple, List, Union, Dict, Optional

import plotly.graph_objects as go


class kNN_forecast:
    def __init__(self, pca_comp: int = 35, n_nbr: int = 5):
        # Hyperparameters
        self.pca_comp: int = pca_comp               # Number of PCA components to use
        self.n_nbr: int = n_nbr                     # Number of neighbors in kNN

        # Time windows
        self.train_window: Optional[pd.DatetimeIndex] = None   # Training date range
        self.predict_window: Optional[pd.DatetimeIndex] = None # Prediction date range

        # Input data
        self.train_df: Optional[pd.DataFrame] = None           # Training data
        self.predict_df: Optional[pd.DataFrame] = None         # Data for prediction

        # Prediction output
        self.predicted_values: Optional[List[float]] = None    # Final predicted values
        self.true_vals_if_forcast_within_downloaded_data: Optional[Dict[str, float]] = None  
                                                               # Actuals, if forecast falls within known data

        # Data availability cutoff
        self.data_already_downloaded_till: pd.Timestamp = pd.Timestamp('2025-06-20')  
                                                               # Latest available data


    def plot_plotly(self) -> go.Figure:
        """
        Plots historical (true values), then extends on the same plot forecasted daily power 
        output over a time range using Plotly.

        Args:
            preds (List[float]): Predicted daily power output values.
            trained_on (List[float]): Historical aggregate power values used for training.
            train_window (pd.DatetimeIndex): Timestamps corresponding to training data.
            predict_window (pd.DatetimeIndex): Timestamps corresponding to prediction days.

        Returns:
            fig (go.Figure): A plotly figure of the true and predicted values 
        """
        fig = go.Figure()

        trained_on = np.array(self.train_df.wind)
        trained_on = [sum(trained_on[i:i+24]) for i in range(0,len(trained_on),24)]

        preds = self.predicted_values
        train_window = self.train_window
        predict_window = self.predict_window

        predict_day_strings = [train_window[-1]] + list(predict_window)
        preds = [trained_on[-1]] + preds

        if self.predict_window[-1] <= self.data_already_downloaded_till:
            ## if forecast date range lies inside the downloaded data
            ## that means, we have power data for this range. In this case,
            ## we also plot the true values
            fig.add_trace(go.Scatter(
                x=predict_day_strings,
                y=[trained_on[-1]] + self.true_vals_if_forcast_within_downloaded_data,
                mode='lines',
                name='True values',
                line=dict(color='cyan')
            ))

        fig.add_trace(go.Scatter(
            x=train_window,
            y=trained_on,
            mode='lines',
            name='Trained data',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=predict_day_strings,
            y=preds,
            mode='lines',
            name='Predicted',
            line=dict(color='magenta', dash='dash')
        ))

        fig.update_layout(
            title="Plot of total power per day showing the trained data and the predicted data.",
            xaxis_title='Date',
            yaxis_title='Total power per day (in MW)',
            xaxis=dict(
                tickformat='%Y-%m-%d',
                tickangle=45,
                tickmode='array',
                tickvals=pd.date_range(start=train_window[0], end=predict_day_strings[-1], periods=12)
            ),
            legend=dict(x=0.01, y=0.99)
        )

        return fig



    def list_of_days(self, time_window: pd.DatetimeIndex) -> List[str]:
        """
        Provides a list of string-formatted dates (YYYY-MM-DD) from a given DatetimeIndex.

        Args:
            time_window (pd.DatetimeIndex): Array of pandas timestamps.

        Returns:
            List[str]: List of dates in 'YYYY-MM-DD' string format.
        """
        assert isinstance(time_window, pd.DatetimeIndex), ("TypeError: "
                                    "time_window must be of type pandas.core.indexes.datetimes.DatetimeIndex.")
        return [x.strftime('%Y-%m-%d') for x in time_window]


    def speed_cubed_div_temp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering step: Computes (wind_speed^3) / (temperature in Kelvin) for each farm.

        Args:
            df (pd.DataFrame): Input DataFrame with wind speed and temperature columns for 39 farms.

        Returns:
            pd.DataFrame: Modified DataFrame with new features added and original weather columns dropped.
        """    
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


    def read_and_clean_power_data(self) -> pd.DataFrame:
        """
        Reads historical wind power data and filters it to match the training window.
        Automatically fetches new data from API if needed and handles duplicate timestamps.

        Returns:
            pd.DataFrame: Cleaned DataFrame containing 'time' and 'wind' columns.
        """
        train_window = self.train_window
        file_path = os.path.join(os.getcwd(), "already_downloaded_data\\power_data.csv")
        power_df = pd.read_csv(file_path)[["time","wind"]]

        power_df['time'] = pd.to_datetime(power_df['time'])
        power_df['time'] = power_df['time'].apply(lambda x: x.floor('h'))
        power_df['time'] = power_df['time'].dt.tz_localize(None)
        power_df = power_df.sort_values('time')

        if self.predict_window[-1] <= self.data_already_downloaded_till:
            ## if forecast window falls under the downloaded range, also save the the true values to plot
            predict_window_hourly = pd.date_range(self.predict_window[0],
                                            self.predict_window[-1].replace(hour=23), freq='h')
            predict_window_power = power_df[power_df['time'].isin(predict_window_hourly)]
            pw_power_hourly = np.array(predict_window_power.wind)
            pw_power_daily = [sum(pw_power_hourly[i:i+24]) for i in range(0,len(pw_power_hourly),24)]
            self.true_vals_if_forcast_within_downloaded_data = pw_power_daily

        train_window_hourly =  pd.date_range(train_window[0], 
                                                train_window[-1].replace(hour=23), freq='h')
        power_df_train = power_df[power_df['time'].isin(train_window_hourly)]
   
        
        if (len(power_df_train[power_df_train['time'].duplicated()])  != 0
            or len(power_df_train[power_df_train.isnull().any(axis=1)]) != 0) :
            print("Duplicates or NaN found in power data. Processing and cleaning data...")
            power_df_train = self.duplicate_NaN_handle(power_df_train)

        if train_window[-1] <= self.data_already_downloaded_till: 
                                        ## if the predict window is before 'data_already_downloaded_till'
                                        ## we'll use the already downloaded data to train and predict.
                                        ## This will improve the runtime significantly as most of the data
                                        ## will already be downloaded apart from a few days
                                                            
            return power_df_train
        else:
            start = self.data_already_downloaded_till + pd.Timedelta(1, unit='d')
            end = train_window[-1].strftime('%Y-%m-%d')            
            new_rows =  self.power_api_call(start,end)
            new_rows['time'] = pd.to_datetime(new_rows['time'], utc =True)
            new_rows['time'] = new_rows['time'].apply(lambda x: x - pd.Timedelta(hours=5))
            new_rows['time'] = new_rows['time'].apply(lambda x: x.floor('h'))
            new_rows['time'] = new_rows['time'].dt.tz_localize(None)

            new_power_df = pd.merge(power_df_train, new_rows, on=['time'])
            if (len(new_power_df[new_power_df['time'].duplicated()]) != 0
                or len(new_power_df[new_power_df.isnull().any(axis=1)])):
                print("Duplicates or NaN found in downloaded power data. Processing and cleaning data...")
                new_power_df = self.duplicate_NaN_handle(new_power_df)
            return new_power_df

    
    def duplicate_NaN_handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles duplicate timestamps in the power data.

        Args:
            df (pd.DataFrame): Power data with potential duplicate timestamps.

        Returns:
            pd.DataFrame: DataFrame with duplicates resolved.
        """
        df = df.set_index('time').sort_index()
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
        
        df = df[~df.index.duplicated(keep='first')]
        df = df.reindex(full_index)

        df = df.ffill()
        df = df.bfill()

        return df.reset_index().rename(columns={'index': 'time'})


    def power_api_call(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches wind generation data from the Hydro Quebec API for the given date range.

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: DataFrame containing hourly wind power data.
        """ 
        api_key = 'API_key_here'
        data_type = 'generation'
        data_frame = Hydro_quebec_data(api_key, data_type, start_date, end_date)
        return data_frame ### may need to convert to pd.Dataframe as the api call may return a dictionary


    def read_and_clean_weather_data(self)-> pd.DataFrame:
        """
        Reads and preprocesses weather data for both training and prediction windows. 
        Automatically fetches new data from API if needed and handles duplicate timestamps.

        Returns:
            pd.DataFrame: Cleaned and complete weather data for the combined time window.
        """
        file_path = os.path.join(os.getcwd(), "already_downloaded_data\\weather_data.csv")
        weather_df = pd.read_csv(file_path)

        train_window = self.train_window
        predict_window = self.predict_window

        train_window = self.list_of_days(train_window)
        predict_window = self.list_of_days(predict_window)
        time_window = pd.to_datetime(train_window + predict_window)

        weather_df['time'] = pd.to_datetime(weather_df['time'])
        weather_df['time'] = weather_df['time'].dt.tz_localize(None)
        weather_df['time'] = weather_df['time'].apply(lambda x: x.floor('h'))

        time_window_hourly = pd.date_range(time_window[0], time_window[-1].replace(hour=23), freq='h')
        weather_df = weather_df[weather_df['time'].isin(time_window_hourly)]
        weather_df = weather_df.sort_values('time')

        if (len(weather_df[weather_df['time'].duplicated()])  != 0
            or len(weather_df[weather_df.isnull().any(axis=1)]) != 0) :
            print("Duplicates or NaN found in weather data. Processing and cleaning data...")
            weather_df = self.duplicate_NaN_handle(weather_df)

        if time_window[-1] <= self.data_already_downloaded_till: ## if the predict window is before '2025-06-20'
                                                            ## we'll use the already downloaded data to train and predict.
                                                            ## This will improve the runtime significantly as most of the data
                                                            ## will already be downloaded apart from a few days 
            return weather_df
        else:           ## if some rows don't exist, download them, merge and return the new dataframe
            start = self.data_already_downloaded_till + pd.Timedelta(1,unit='d') 
            end = time_window[-1].strftime('%Y-%m-%d')
            new_rows = weather_api_call(start, end)
            new_rows['time'] = pd.to_datetime(new_rows['time'], utc =True)
            new_rows['time'] = new_rows['time'].apply(lambda x: x - pd.Timedelta(hours=5))
            new_rows['time'] = new_rows['time'].apply(lambda x: x.floor('h'))
            new_rows['time'] = new_rows['time'].dt.tz_localize(None)

            new_weather_df = pd.merge(weather_df, new_rows, on= ['time'])
            if (len(new_weather_df[new_weather_df['time'].duplicated()]) != 0
                or len(new_weather_df[new_weather_df.isnull().any(axis=1)])):
                print("Duplicates or NaN found in downloaded weather data. Processing and cleaning data...")
                new_weather_df = self.duplicate_NaN_handle(new_weather_df)
            return new_weather_df

        
    def input(self) -> None:
        """
        Combines cleaned power and weather data for both training and prediction periods.
        Finally updates variables self.train_df and self.predict_df

        Returns:
            None
        """
        self.get_forecast_window()
        self.get_train_window()

        power_df = self.read_and_clean_power_data()          ## already only has the training data
        weather_df = self.read_and_clean_weather_data()
        
        weather_df_train = weather_df[weather_df['time'].isin(pd.date_range(self.train_window[0], 
                                                self.train_window[-1].replace(hour=23), freq='h'))]
        self.train_df = pd.merge(power_df, weather_df_train, on=['time'])

        self.predict_df= weather_df[weather_df['time'].isin(pd.date_range(self.predict_window[0], 
                                                self.predict_window[-1].replace(hour=23), freq='h'))]    
                    ## assuming in real-time case, we won't have the wind power column in the predict_df
                    ## that's the entire point that we are predicting anyway


    ## This one is for the dash app input
    def set_input(self, start_date: str, forecast_window: int) -> str:
        """
        To set the input data coming from the dash app.
        Combines cleaned power and weather data for both training and prediction periods.
        Finally updates variables self.train_window, self.train_df and self.predict_df.

        Args:
            start_date (str): Start date from the prediction window in YYYY-MM-DD format.
            forecast_window (int): Forcast window length.

        Returns:
            str: Error messages for different situations
        """
        try:
            start_date = pd.to_datetime(start_date)
            earliest_allowed_date = pd.Timestamp('2024-08-20')
            latest_allowed_date = pd.Timestamp(str(date.today()))

            if not (earliest_allowed_date <= start_date <= latest_allowed_date):
                return (f"Error: Please input a date between "
                        f"{earliest_allowed_date.strftime('%Y-%m-%d')} and "
                        f"{latest_allowed_date.strftime('%Y-%m-%d')}.")

            window_length = 3
            if not (1 <= forecast_window <= window_length):
                return f"Error: Forecast window should be an integer between 1 and {window_length}."

            # If valid, compute predict_window
            end_date = start_date + pd.Timedelta(days=forecast_window - 1)
            self.predict_window = pd.date_range(start=start_date, end=end_date, freq='d')
            self.get_train_window()
            power_df = self.read_and_clean_power_data()          ## already only has the training data
            weather_df = self.read_and_clean_weather_data()
            
            weather_df_train = weather_df[weather_df['time'].isin(pd.date_range(self.train_window[0], 
                                                    self.train_window[-1].replace(hour=23), freq='h'))]
            self.train_df = pd.merge(power_df, weather_df_train, on=['time'])

            self.predict_df= weather_df[weather_df['time'].isin(pd.date_range(self.predict_window[0], 
                                                    self.predict_window[-1].replace(hour=23), freq='h'))]  

            # Return None or success message
            return ""


        except Exception as e:
            return f"Unexpected error: {str(e)}"


    def get_forecast_window(self) -> None:
        """
        Interactively prompts the user to input a forecast window. Used when run from a terminal
        or jupyter notebook.

        """
        earliest_allowed_date = pd.Timestamp('2024-08-20')
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

        print(f"\nYou have entered: {window}")

        end_date = start_date + pd.Timedelta(window-1, unit='d')
        predict_window = pd.date_range(start=start_date, end=end_date, freq='d')
        
        print(f"\nYour forecast window is days between: {predict_window[0].strftime('%Y-%m-%d')} "
            f"and {predict_window[-1].strftime('%Y-%m-%d')}\n")
        self.predict_window = predict_window


    def get_train_window(self) -> pd.DatetimeIndex:
        """
        Computes the time window to train on: 60 days before the forecast window, 
        or starting from 2019-01-01 if the forecast is too early.
        
        Returns:
            train_window (pd.DatetimeIndex): The training dataset window.
        """
        days_to_train_on = 60
        predict_start_date = self.predict_window[0]
        time_diff = predict_start_date - pd.Timestamp(year=2019,month=1,day=1)
        if time_diff.days <= days_to_train_on:
            train_start_date = pd.Timestamp(year=2019,month=1,day=1)
        else:
            train_start_date = predict_start_date - pd.Timedelta(days_to_train_on, unit='d')
        self.train_window = pd.date_range(start= train_start_date, 
                                    end= predict_start_date - pd.Timedelta(1, unit='d'), freq='d')   


    def forecast(self) -> go.Figure:
        """
        Runs the kNN pipeline on weather and power data: reads, preprocesses, trains, predicts, and plots.

        Returns:
            Tuple[List[float], go.Figure]: Daily forecasted values and a plot.

        """

        print("Please wait. Plotting your forecast...")
        train_df = self.speed_cubed_div_temp(self.train_df)
        predict_df = self.speed_cubed_div_temp(self.predict_df)

        features = train_df.columns.drop(['Unnamed: 0.1','Unnamed: 0', 'time', 'Year', 'YearMonthDay', 'wind'], 
                                        errors='ignore')

        pipe = Pipeline([('scale', StandardScaler()), ('pca', PCA(n_components= self.pca_comp))])
        pipe.fit(train_df[features])

        pca_train = pipe.transform(train_df[features])
        pca_predict = pipe.transform(predict_df[features])

        knn = KNeighborsRegressor(n_neighbors=self.n_nbr)
        knn.fit(pca_train, train_df.wind)
        pred = knn.predict(pca_predict)

        predicted = [sum(pred[i:i+24]) for i in range(0, len(pred),24)] 
        self.predicted_values = predicted 

        fig = self.plot_plotly()       
        fig.show()
        return fig
