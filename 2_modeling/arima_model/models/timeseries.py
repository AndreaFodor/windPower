import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import date, datetime, timedelta
from statsforecast.models import AutoARIMA
from statsforecast import StatsForecast
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from seaborn import set_style
set_style("whitegrid")

def load_training_df(path='../../1_data/final_dataframes/main_validation_dataframe.csv'):
    """
    Load the training dataframe and aggregate the data by day.

    Parameters
    ----------
    path : str
        path to the training dataframe

    Returns
    -------
    str
        training dataframe
    """
    try:
        df = pd.read_csv(path)
    except:
        raise Exception('File not found!')
    
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])
    coldict = {}
    for col in numeric_cols.columns:
        if col=='Wind':
            coldict[col] = 'sum'
        else:
            coldict[col] = 'mean'

    daily = numeric_cols.resample('D').agg(coldict)

    daily['day_of_year'] = daily.index.dayofyear
    daily['year'] = daily.index.year
    daily['month'] = daily.index.month

    nfarms = 0
    for column in df.columns:
        if 'temperature' in column:
            nfarms += 1
            
    daily['mean_temperature'] = daily[[f'temperature_2m_{i}' for i in range(1, nfarms)]].mean(axis=1)
    daily['mean_humidity'] = daily[[f'relative_humidity_2m_{i}' for i in range(1, nfarms)]].mean(axis=1)
    daily['mean_windspeed'] = daily[[f'wind_speed_10m_{i}' for i in range(1, nfarms)]].mean(axis=1)

    return daily


class ARIMAxPredictions():
    """
    Model for ARIMA predictions with exogenous variables; 
    uses past 60 days of wind power generation and mean windspeed to predict 
    the wind power generation for the next day
    """
    def __init__(self):
        self.horizon = 1
        self.training_interval = 60
        self.confidence_level = [90]
        self.models = [AutoARIMA()]
        self.df_train = None
        self.forecastarr = []

    def load_training_data(self, path='../../1_data/final_dataframes/main_validation_dataframe.csv', datetime_column='index', predict_column='Wind', exogenus_column='mean_windspeed'):
        """
        path: path to training dataframe, needs to have at least 60 days of data available prior to prediction date
        datetime_column: name of the column in df containing timestamps (can be 'index')
        predict_column: name of the column in df containing wind power values
        exogenous_column: name of the column in df containing mean windspeed values

        """
        df = load_training_df()
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df must be a pd.DataFrame')
        if not (datetime_column in df.columns and predict_column in df.columns and exogenus_column in df.columns):
            if (datetime_column == 'index' and predict_column in df.columns and exogenus_column in df.columns):
                pass
            else:
                raise Exception('df needs to contain datetime_column, predict_column and exogenous_column')        

        df_train = pd.DataFrame()
        if datetime_column == 'index':
            df_train['ds'] = df.index
        else:
            df_train['ds'] = df[datetime_column]
        df_train['unique_id'] = 'A'
        df_train['y'] = df[predict_column].values
        df_train['windspeed'] = df[exogenus_column].values

        self.df_train = df_train

    def forecast(self, date):
        """
        date: date for which to predict the wind power generation; pd.Timestamp

        Returns:
            fcst: dataframe containing the forecast and confidence intervals
        """
        if not isinstance(date, pd.Timestamp):
            try:
                date = pd.Timestamp(date)
            except:
                raise TypeError('date must be a string or pd.Timestamp')
        
        if not date in self.df_train['ds'].values:
            raise Exception('df has to contain the desired date')
        if len(self.df_train[(self.df_train['ds']<date) & (self.df_train['ds']>(date - pd.Timedelta(days=self.training_interval + 1)))])<self.training_interval:
            raise Exception('df does not contain all the training dates; either provide a df with more dates or change self.training_interval')
        
        if self.df_train is None:
            self.load_training_data()

        stfcst = StatsForecast(models=self.models, 
                           freq='D', 
                           n_jobs=-1,
                           )
        fcst = stfcst.forecast(df=self.df_train[(self.df_train['ds']<date) & (self.df_train['ds']>(date - pd.Timedelta(days=self.training_interval + 1)))], 
                           h=self.horizon, 
                           X_df=self.df_train.drop(columns='y')[self.df_train['ds']==date],
                           level=self.confidence_level,
                           )
        return fcst
    
    def forecast_range(self, start_date, end_date):
        if not isinstance(start_date, pd.Timestamp):
            try:
                start_date = pd.Timestamp(start_date)
            except:
                raise TypeError('start_date must be a string or pd.Timestamp')
        if not isinstance(end_date, pd.Timestamp):
            try:
                end_date = pd.Timestamp(end_date)
            except:
                raise TypeError('end_date must be a string or pd.Timestamp')
        if not end_date > start_date:
            raise Exception('End date must be after start date!')
        
        if self.df_train is None:
            self.load_training_data()
        for date in pd.date_range(start=start_date, end=(end_date), freq='D'):
            fcst = self.forecast(date)
            self.forecastarr.append(fcst.AutoARIMA.values)
        cmape = mape(self.df_train[(self.df_train['ds']>=start_date) & (self.df_train['ds']<=end_date)].y.values, self.forecastarr)
        cmae = mae(self.df_train[(self.df_train['ds']>=start_date) & (self.df_train['ds']<=end_date)].y.values, self.forecastarr)
        cr2 = r2_score(self.df_train[(self.df_train['ds']>=start_date) & (self.df_train['ds']<=end_date)].y.values, self.forecastarr)
        print('Mean absolute percentage error: ', cmape)  
        print('Mean absolute error: ', cmae)
        print('r2 score: ', cr2)  

        plt.plot(self.df_train[(self.df_train['ds']<=start_date) & (self.df_train['ds']>=(start_date - pd.Timedelta(days=self.training_interval + 1)))].ds.values, self.df_train[(self.df_train['ds']<=date) & (self.df_train['ds']>=(date - pd.Timedelta(days=self.training_interval + 1)))].y.values, label = 'training data')
        plt.plot(self.df_train[(self.df_train['ds']>=start_date) & (self.df_train['ds']<=end_date)].ds.values, self.df_train[(self.df_train['ds']>=start_date) & (self.df_train['ds']<=end_date)].y.values, label = 'True value')
        plt.plot(self.df_train[(self.df_train['ds']>=start_date) & (self.df_train['ds']<=end_date)].ds.values, self.forecastarr, label = 'Prediction')
        plt.xlabel('Date')
        plt.ylabel('Wind power per day [MW]')
        plt.xticks(rotation=45)
        plt.legend()

        self.plot_plotly(start_date, end_date)

        return cmape, cr2


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
            self.horizon = pd.date_range(start=start_date, end=end_date, freq='d')
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


    def plot_plotly(self, start_date, end_date) -> go.Figure:
        fig = go.Figure()
        
        date_range = pd.date_range(start=start_date, end=(end_date), freq='D')
        trained_on = self.df_train[(self.df_train['ds']>=start_date) & (self.df_train['ds']<=end_date)].y.values

        # if self.predict_window[-1] <= self.data_already_downloaded_till:
        #     ## if forecast date range lies inside the downloaded data
        #     ## that means, we have power data for this range. In this case,
        #     ## we also plot the true values
        #     fig.add_trace(go.Scatter(
        #         x=date_range,
        #         y=[trained_on[-1]] + self.true_vals_if_forcast_within_downloaded_data,
        #         mode='lines',
        #         name='True values',
        #         line=dict(color='cyan')
        #     ))

        fig.add_trace(go.Scatter(
            x=date_range,
            y=trained_on,
            mode='lines',
            name='Trained data',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=date_range,
            y=self.forecastarr,
            mode='lines',
            name='Predicted',
            line=dict(color='magenta', dash='dash')
        ))

        fig.update_layout(
            title="Plot of total power per day showing the training data and the predicted data.",
            xaxis_title='Date',
            yaxis_title='Total power per day (in MW)',
            xaxis=dict(
                tickformat='%Y-%m-%d',
                tickangle=45,
                tickmode='array',
                # tickvals=pd.date_range(start=train_window[0], end=predict_day_strings[-1], periods=12)
            ),
            legend=dict(x=0.01, y=0.99)
        )
        fig.show()

        return fig

