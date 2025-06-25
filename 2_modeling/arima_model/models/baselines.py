import numpy as np
import pandas as pd

class NaiveBaseline:
    def __init__(self):
        self.last_value = None

    def fit(self, ts):
        """Fit the model by storing the last value of the time series.
        Args:
            ts (numpy array or pandas Series): The time series data.
        """
        # Convert pandas Series to numpy array if input is a pandas Series
        if isinstance(ts, pd.Series):
            ts = ts.values  # Convert to numpy array
        self.last_value = ts[-1]
        
    def forecast(self, h):
        """Forecast for the next h time periods using the last observed value.
        Args:
            h (int): Number of periods to forecast.
        Returns:
            numpy array: Forecasted values.
        """
        return np.full(h, self.last_value)

class RandomWalkWithDrift:
    '''
    Remember that the Random Walk With Drift Model leads to a forecast which 
    simply extends the secant line connecting the first and last point of the time series.
    '''
    def __init__(self):
        self.initial_value = float('nan')  # Initialize as NaN to indicate unfit state
        self.last_value = float('nan')
        self.slope = float('nan')

    def fit(self, ts):
        """Fit the model to the time series data."""
        # Convert pandas Series to numpy array if input is a pandas Series
        if isinstance(ts, pd.Series):
            ts = ts.values  # Convert to numpy array
        
        # Check if the series has at least two points
        if len(ts) < 2:
            raise ValueError("Time series must have at least two data points.")
        
        self.initial_value = ts[0]
        self.last_value = ts[-1]
        self.slope = (ts[-1] - ts[0]) / (len(ts) - 1)  # Slope of the line

    def forecast(self, h):
        """Generate a forecast for the next 'h' time periods."""
        # Check if model has been fitted
        if np.isnan(self.last_value):
            raise ValueError("The model has not been fitted yet. Call 'fit' first.")
        
        # Generate forecast
        return self.last_value + self.slope * np.arange(1, h + 1)

class NaiveSeasonalWithDrift:
    """
    Forecasts future values by repeating the most recent seasonal
    pattern of differences (deltas).

    For example, if season_length = 4 and the last 5 values were:
    [3,4,5,6,5], then deltas would be [1,1,1,-1] and the last_value would be 5.

    The forecast into the future would be [6,7,8,7, 8, 9, 10, 9,...].
    """
    def __init__(self, season_length):
        self.season_length = season_length
        self.last_season = None
        self.deltas = None
        self.last_value = None

    def fit(self, ts):
        if isinstance(ts, pd.Series):
            ts = ts.values
        s = self.season_length
        if len(ts) < s + 1:
            raise ValueError("Need more data than one season")
        self.deltas = ts[-s:] - ts[-s-1:-1]
        self.last_value = ts[-1]

    def forecast(self, h):
        s = self.season_length
        forecasts = np.zeros(h)
        forecasts[0] = self.last_value + self.deltas[0]
        for i in range(1, h):
            forecasts[i] = forecasts[i-1]+self.deltas[i%s]
        return forecasts