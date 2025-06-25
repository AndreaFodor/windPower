import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
from typing import Tuple, List, Union, Dict, Optional



class kNN_Cross_Validation:
    def __init__(
        self, 
        pca_comps: List[int] = [35],
        n_nbrs: List[int] = [5],
        days_to_train_on: int = 60,
        mode: str = "Validation",
        feature_selection: str = "spd_cubed_div_temp",
        with_confidence_interval: bool = False,
        no_iters: int = 50,
        path: str = None
    ):
        # Hyperparameters
        self.path = path
        hyperparam_list_len = 5
        assert ((len(pca_comps) <= hyperparam_list_len) and 
                (len(n_nbrs) <=hyperparam_list_len)), ("\nPlease provide the lists of pca components"
                f" and kNN neighbours of length less than {hyperparam_list_len} to reduce run-time.")
        
        if with_confidence_interval:
            assert ((len(pca_comps) <= 1) and (len(n_nbrs) <=1)), ("\n To reduce run-time, confidence "
            "interval calculation is implented only with one hyperparameter combination pair. "
            "Please provide single hyperparameter pair (pca_comp, n_nbr).")
            assert (5 <= no_iters <= 100), ("\n To reduce run-time, number of iterations is limited"
            "to a value between 5 and 100.")
            self.no_iters = no_iters
        self.ci = with_confidence_interval
        self.pca_comps: List[int] = pca_comps  # List of PCA components to try
        self.n_nbrs: List[int] = n_nbrs        # List of kNN neighbor counts to try


        assert (10 <= days_to_train_on <= 1095), ("Please input training window "
                        "length (in days) to be an integer between 10 and 1095")
        self.days_to_train_on: int = days_to_train_on  # Days to use for training in each CV fold
        assert (mode in ["Validation","Testing"]), ("Parameter mode must be in "
                                                       " [\"Validation\" or \"Testing\"].")
        self.mode: str = mode  # Either "Validation" or "Testing"
        assert (feature_selection in ["spd_cubed_div_temp", "speed", "speed_cubed"]), ("Parameter"
        " feature_selection must be in [\"spd_cubed_div_temp\", \"speed\", \"speed_cubed\"].")
        self.feature_selection: str = feature_selection 
                        # Either "spd_cubed_div_temp", "speed", "speed_cubed"

        # Prediction/test window (to be set later by the user)
        self.test_window: Optional[pd.DatetimeIndex] = None

        # Used when there's only a single (PCA, kNN) combination
        self.preds: Optional[Union[List[float], List[List[float]]]] = None  #List[List[float]] in case with_ci is True
        self.true_aggregates: Optional[Dict[str, float]] = None
        self.mape: Optional[float] = None
        self.r2: Optional[float] = None
        self.mae: Optional[float] = None

        self.pred_conf_intervals: Optional[List[(float,float)]] = None
        self.mape_conf_intervals: Optional[Tuple[float,float]] = None
        self.r2_conf_intervals: Optional[Tuple[float,float]] = None
        self.mae_conf_intervals: Optional[Tuple[float,float]] = None

        self.figure: Optional[go.Figure] = None

        # Used when testing multiple hyperparameter combinations
        self.hyperparam_comb_index: Optional[Dict[int, Tuple[int, int]]] = None  # List of (pca_idx, kNN_idx)
        self.hyperparam_comb_preds: Optional[List[List[float]]] = None      # Preds for each hyperparam combo
        self.cv_mapes: Optional[List[float]] = None                         # MAPE scores across CV folds
        self.cv_r2s: Optional[List[float]] = None                           # R² scores across CV folds
        self.cv_maes: Optional[List[float]] = None                             # MAE scores across CV folds
        self.best_MAPE_idx: Optional[int] = None                            # Index of best MAPE result
        self.best_r2_idx: Optional[int] = None                              # Index of best R² result
        self.best_mae_idx: Optional[int] = None                             # Index of beest MAE result

        # Plotly figures for best results
        self.fig_best_MAPE: Optional[go.Figure] = None     # For best combination in terms ofMAPE
        self.fig_best_r2: Optional[go.Figure] = None       # For best combination in terms of R²
        self.fig_best_mae: Optional[go.Figure] = None      # For best combination in terms of MAE
        

    def list_of_days(self, time_window: pd.DatetimeIndex) -> List[str]:
        """
        Converts a DatetimeIndex into a list of day strings in 'yyyy-mm-dd' format.

        Args:
            time_window (pd.DatetimeIndex): A pandas DatetimeIndex representing the time window.

        Returns:
            list of str: A list of dates in 'yy-mm-dd' format.
        """
        assert isinstance(time_window, pd.DatetimeIndex), ("TypeError: "
                                    "time_window must be of type pandas.core.indexes.datetimes.DatetimeIndex.")
        return [x.strftime('%Y-%m-%d') for x in time_window]


    def plot(self, preds: Union[List[float], List[List[float]]], true: Dict[str, float], 
             mape: float, r2: float, mae: float, pca_comp: int, n_nbr: int) -> go.Figure:
        """
        Returns a Plotly figure of true vs predicted values for a prediction time window,
        annotated with MAPE and R² scores and model parameters.

        Args:
            preds (List[float]): List of predicted values.
            true (Dict[str, float]): Dictionary with date as key and the true 
                                     aggregate power for the day as value.
            mape (float): Mean Absolute Percentage Error.
            r2 (float): R² score.
            mae (float): MAE score
            pca_comp (int): PCA component used for the model.
            n_nbr (int): Number of kNN neighbours used for the model. 

        Returns:
            go.Figure: Plotly graph object representing the forecast plot.
        """
        days = list(true.keys())
        true_vals = list(true.values())

        fig = go.Figure()

        # True values trace
        fig.add_trace(go.Scatter(
            x=days,
            y=true_vals,
            mode='lines',
            name='True',
            line=dict(color='blue')
        ))

        # Predicted values trace
        if self.ci:
            preds = np.array(preds)
            mean_preds = np.mean(preds, axis=1)
            lower = [x for (x,y) in self.pred_conf_intervals]
            upper = [y for (x,y) in self.pred_conf_intervals]
            # CI lower bound (invisible)
            fig.add_trace(go.Scatter(
                x=days,
                y=lower,
                mode='lines',
                name='Lower Bound',
                line=dict(color='lightblue'),
                showlegend= False
            ))

            # CI upper bound (filled to previous trace)
            fig.add_trace(go.Scatter(
                x=days,
                y=upper,
                mode='lines',
                name='95% CI',
                line=dict(color='lightblue'),
                fill='tonexty',
                fillcolor='rgba(173, 216, 230, 0.3)',  
            ))

            # Mean prediction
            fig.add_trace(go.Scatter(
                x=days,
                y=mean_preds,
                mode='lines',
                name='Predicted (mean)',
                line=dict(color='cyan', dash='dash')
            ))

        else:   # single prediction vector
            fig.add_trace(go.Scatter(
                x=days,
                y=preds,
                mode='lines',
                name='Predicted',
                line=dict(color='cyan', dash='dash')
            ))

        # Annotated box as text annotation
        annotation_text = f"PCA: {pca_comp} | kNN nbr: {n_nbr} | MAPE: {mape:.3f} | R²: {r2:.3f} | MAE: {mae:.1f}"
        
        fig.add_annotation(
            text=annotation_text,
            xref="paper", yref="paper",
            x=0.5, y=0.95,
            showarrow=False,
            font=dict(size=12),
            align='left',
            bgcolor="wheat",
            opacity=0.6
        )

        # X ticks: subsample like matplotlib version
        tick_count = min(12, len(days))
        tick_vals = [days[i] for i in range(0, len(days), max(1, len(days)//tick_count))]

        fig.update_layout(
            title="Plot of total power per day showing true and predicted.",
            xaxis=dict(
                title="Date",
                tickangle=45,
                tickvals=tick_vals
            ),
            yaxis=dict(title="Total power per day (in MW)"),
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=40, r=20, t=60, b=60)
        )

        return fig


    def make_yymmdd_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a 'YearMonthDay' column in 'yy-mm-dd' format derived from the 'time' column.
        Drops the 'MonthDay' column if present.

        Args:
            df (pd.DataFrame): DataFrame with a 'time' column and a 'MonthDay' column.

        Returns:
            pd.DataFrame: Modified DataFrame with 'YearMonthDay' column added and 'MonthDay' removed.
        """
        try:
            new_df = df
            new_df.time = pd.to_datetime(new_df.time).dt.tz_localize(None)
            new_df['YearMonthDay'] = new_df['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
            new_df = new_df.drop(columns = ['MonthDay'])
            return new_df
        except:
            print("Incorrect dataframe passed to make_yymmdd_format.")

    def speed_cubed_div_temp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates (wind speed)^3 divided by temperature (in Kelvin) for each farm
        and adds these as new columns. Drops the original weather columns.

        Args:
            df (pd.DataFrame): Input DataFrame with wind speed and temperature columns.

        Returns:
            pd.DataFrame: Modified DataFrame with new features and dropped originals.
        """  
        new_cols = []
        for i in range(1, 40):
            spd_col = f'wind_speed_10m_{i}'
            temp_col = f'temperature_2m_{i}'
            final_col = (df[spd_col] ** 3)/ (df[temp_col].values + 273)     
            new_cols.append(pd.DataFrame({
                f'{spd_col}_cubed_div_temp': final_col
            }))

        df_new = pd.concat([df.drop(columns= [f'wind_speed_10m_{i}' for i in range(1, 40)] +
                                    [f'temperature_2m_{i}' for i in range(1,40)] +
                                    [f'wind_direction_10m_{i}' for i in range(1,40)]+
                                    [f'relative_humidity_2m_{i}' for i in range(1,40)], errors='ignore')] 
                                    + new_cols, axis=1)
        return df_new.copy()

    def speed_cubed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates (wind speed)^3 for each farm and adds these as new columns. 
        Drops the original weather columns.

        Args:
            df (pd.DataFrame): Input DataFrame with wind speed and temperature columns.

        Returns:
            pd.DataFrame: Modified DataFrame with new features and dropped originals.
        """  
        new_cols = []
        for i in range(1, 40):
            spd_col = f'wind_speed_10m_{i}'
            final_col = df[spd_col] ** 3
            
            new_cols.append(pd.DataFrame({
                f'{spd_col}_cubed': final_col
            }))
        df_new = pd.concat([df.drop(columns= [f'wind_speed_10m_{i}' for i in range(1, 40)] +
                                    [f'temperature_2m_{i}' for i in range(1,40)] +
                                    [f'wind_direction_10m_{i}' for i in range(1,40)]+
                                    [f'relative_humidity_2m_{i}' for i in range(1,40)], errors='ignore')] 
                                    + new_cols, axis=1)
        return df_new.copy()
    


    def true_aggregate_per_day(self, df: pd.DataFrame, time_window: pd.DatetimeIndex) -> Dict[str, float]:
        """
        Computes (true) total wind power per day over a given time window.

        Args:
            df (pd.DataFrame): DataFrame with 'YearMonthDay' and 'Wind' columns.
            time_window (pd.DatetimeIndex): Range of dates to compute aggregates for.

        Returns:
            Dict[str, float]: A dictionary mapping 'yy-mm-dd' strings to total wind production values.
        """
        try:
            assert isinstance(time_window, pd.DatetimeIndex), ("TypeError: "
                    "time_window must be of type pd.DatetimeIndex.")
            day_list = self.list_of_days(time_window)
            df_test = df[df.YearMonthDay.isin(day_list)]
            aggregates = {}
            for day in day_list:
                df_day = df_test[df_test['YearMonthDay'] == day]
                aggregates[day] = sum(np.array(df_day.Wind))
            return aggregates 
        except:
            print("Incorrect dataframe passed to true_aggregate_per_day.")


    def extract_train_test_data(self, 
            df: pd.DataFrame, train_window: pd.DatetimeIndex, 
            test_window: pd.DatetimeIndex) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the DataFrame into training and testing sets based on the given date range.

        Args:
            df (pd.DataFrame): DataFrame with a 'time' column as pandas datetime.
            train_window (pd.DatetimeIndex): The time window for the training dataframe.
            test_window (pd.DatetimeIndex): The time window for the training dataframe.

        Returns:
            tuple: A tuple (train_df, test_df), where:
                - train_df contains all rows with time in train_window
                - test_df contains all rows with time in test_window.
        """
        try:
            assert (isinstance(train_window, pd.DatetimeIndex) and isinstance(test_window, pd.DatetimeIndex)), (
                "TypeError: start_date and  end_date must be pandas.Timestamp type variable.")
            train_days = self.list_of_days(train_window)
            test_days = self.list_of_days(test_window)
            train_df = df[df['YearMonthDay'].isin(train_days)]
            test_df = df[df['YearMonthDay'].isin(test_days)]
            return train_df, test_df
        except:
            print("Incorrect dataframe passed to extract_train_test_data.")

    def predict_window_validity(self) -> None:
        """
        Validates the format and range of the forecast window.

        Args:
            test_window (pd.DatetimeIndex): Forecast window to validate.

        Raises:
            AssertionError: If the input is not a DatetimeIndex or out of allowed range.
        """
        earliest_allowed_date = (pd.Timestamp(year=2019, month=1, day=1) 
                                 + pd.Timedelta(self.days_to_train_on, unit='d'))
        if self.mode == "Validation":
            latest_allowed_date = pd.Timestamp(year=2022, month=12, day=31)
        else:
            latest_allowed_date = pd.Timestamp(year=2023, month=12, day=31)
        test_window = self.test_window
        assert isinstance(test_window, pd.DatetimeIndex), ("TypeError: test_window must be of type "
                "pd.DatetimeIndex")
        test_window = test_window.sort_values()
        assert (earliest_allowed_date <= test_window[0] 
                and test_window[-1] <= latest_allowed_date), ("ValueError: test_window must be "
                f"between {earliest_allowed_date.strftime('%Y-%m-%d')} and "
                f"{latest_allowed_date.strftime('%Y-%m-%d')}.")
    

    def kNN_on_particular_train_test_splits(self, 
        df_train: pd.DataFrame, df_test: pd.DataFrame, pca_comp: int, n_nbr: int
    ) -> List[float]:
        """
        Fits a kNN regressor model using PCA-reduced training data and returns daily predictions.

        Args:
            df_train (pd.DataFrame): Training DataFrame.
            df_test (pd.DataFrame): Testing DataFrame.
            pca_comp (int): Number of PCA components.
            n_nbr (int): Number of nearest neighbors.

        Returns:
            List[float]: Predicted daily aggregates for the test set.
        """
        features = df_train.columns.drop(['Unnamed: 0.1','Unnamed: 0', 'time', 'Year', 'YearMonthDay', 'Wind'], 
                                        errors='ignore')

        pipe = Pipeline([('scale', StandardScaler()), ('pca', PCA(n_components= pca_comp))])
        pipe.fit(df_train[features])

        pca_train = pipe.transform(df_train[features])
        pca_test = pipe.transform(df_test[features])

        knn = KNeighborsRegressor(n_neighbors=n_nbr)
        knn.fit(pca_train, df_train.Wind)
        pred = knn.predict(pca_test)

        pred_per_day = [sum(pred[i:i+24]) for i in range(0, len(pred),24)]
        return pred_per_day


    def knn_using_particular_hyperparams_and_test_window(self, 
        df: pd.DataFrame, pca_comp: int, n_nbr: int, test_window: pd.DatetimeIndex
    ) -> Union[List[float], List[List[float]]]:
        """
        For each day in the test window, trains a kNN model on historic data leading up to the day
        before the chosen day and predicts wind output for the chosen day. Stores and returns these
        predicted values per day. 

        Args:
            df (pd.DataFrame): Preprocessed DataFrame.
            pca_comp (int): Number of PCA components.
            n_nbr (int): Number of neighbors in kNN.
            test_window (pd.DatetimeIndex): Days to test on.

        Returns:
            Union[List[float], List[List[float]]]:
                - Predictions per day by running the model once per day if self.ci is False (default).
                - All the predictions per day by running the model 50 times per day if self.ci is True
        """
        preds = []
        if self.ci:
            print("Note: Running with iterations... This may take longer time.")

        for test_day in test_window:
            test_day = pd.date_range(test_day,test_day,freq='d') ## pd.Timestamp to pd.DatetimeIndex for function call
            
            train_window = self.get_train_window(test_day)   
            df_train, df_test = self.extract_train_test_data(df, train_window, test_day)
            if not self.ci:
                pred = self.kNN_on_particular_train_test_splits(df_train, df_test, pca_comp, n_nbr)
                preds.append(pred[0])
            else:
                ## to calculate CI for an estimator, for each day, we run the model 50 times with 
                ## each time randomly sampling (with replacement, i.e. bootstrapping) 30 days 
                ## (30 times 24 data points) of data out of training data set df_train for that day.
                preds_for_day = []
                for i in range(self.no_iters):
                    df_train_resampled = df_train.sample(frac=0.5, replace=True)
                    pred_iter = self.kNN_on_particular_train_test_splits(df_train_resampled, 
                                                                         df_test, pca_comp, n_nbr)
                    preds_for_day.append(pred_iter[0])
                preds.append(preds_for_day)               

        return preds

    def get_train_window(self, predict_window: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        Computes the time window to train on, which either starts from 2019-01-01 or starts from 60 days
        prior to predict window.

        Args:
            predict_window (pd.DatetimeIndex): The window to predict on.
        
        Returns:
            train_window (pd.DatetimeIndex): The training dataset window.
        """
        days_to_train_on = self.days_to_train_on
        predict_start_date = predict_window[0]
        time_diff = predict_start_date - pd.Timestamp(year=2019,month=1,day=1)
        if time_diff.days <= days_to_train_on:
            train_start_date = pd.Timestamp(year=2019,month=1,day=1)
        else:
            train_start_date = predict_start_date - pd.Timedelta(days_to_train_on, unit='d')
        train_window = pd.date_range(start= train_start_date, 
                                    end= predict_start_date - pd.Timedelta(1, unit='d'), freq='d')
        return train_window
  

    def input(self) -> None:
        """
        Interactively prompts the user to input a prediction window.
        """
        earliest_allowed_date = (pd.Timestamp(year=2019, month=1, day=1) 
                                 + pd.Timedelta(self.days_to_train_on, unit='d'))
        if self.mode == "Validation":
            latest_allowed_date = pd.Timestamp(year=2022, month=12, day=31)
        else:
            latest_allowed_date = pd.Timestamp(year=2023, month=12, day=31)
        print(f"\nPlease enter forecast start date in the format YYYY-MM-DD." 
            f"The date should be in between {earliest_allowed_date.strftime('%Y-%m-%d')}"
            f" and {latest_allowed_date.strftime('%Y-%m-%d')}.")
        date= input("Input the start date now: ")
        try:
            year, month, day = int(date[:4]), int(date[5:7]), int(date[8:])
            start_date = pd.Timestamp(year=year, month=month, day=day)
            assert (earliest_allowed_date <= start_date <= latest_allowed_date), ("Please input a date"
                    f"between {earliest_allowed_date.strftime('%Y-%m-%d')}"
                    f" and {latest_allowed_date.strftime('%Y-%m-%d')}")
        except ValueError:
            raise AssertionError(f"Invalid date: {year}-{month}-{day}")
        print(f"\nYou have entered: {start_date.strftime('%Y-%m-%d')}.")

        time_diff = latest_allowed_date - start_date 
        window_length = min(365, time_diff.days+1)
        print(f"\nPlease enter forecast window (in days). It should be an integer between 1 or {window_length}.")
        
        window = int(input("Input forecast window: "))
        assert 1 <= window <= window_length, f"ValueError: Input should be an integer between 1 or {window_length}."

        end_date = start_date + pd.Timedelta(window-1, unit='d')
        test_window = pd.date_range(start=start_date, end=end_date, freq='d')

        self.test_window = test_window
        self.predict_window_validity()
        
        print(f"\nYour forecast window is days between: {test_window[0].strftime('%Y-%m-%d')} "
            f"and {test_window[-1].strftime('%Y-%m-%d')}\n")


    def manual_input(self, start_date: str, window:int) -> None:
        """
        Manual user input for the prediction window, otherwise same functionality as input().

        Args:
            start_date (str): Start date of prediction window in the format YYYY-MM-DD
            window (int): Length of the prediction window.
        """
        earliest_allowed_date = (pd.Timestamp(year=2019, month=1, day=1) 
                                 + pd.Timedelta(self.days_to_train_on, unit='d'))
        if self.mode == "Validation":
            latest_allowed_date = pd.Timestamp(year=2022, month=12, day=31)
        else:
            latest_allowed_date = pd.Timestamp(year=2023, month=12, day=31)

        try:    
            start_date = pd.Timestamp(start_date)
            assert (earliest_allowed_date <= start_date <= latest_allowed_date), ("Please input a date"
                    f"between {earliest_allowed_date.strftime('%Y-%m-%d')}"
                    f" and {latest_allowed_date.strftime('%Y-%m-%d')}")
        except ValueError:
            raise AssertionError(f"Invalid date: {start_date}.")

        time_diff = latest_allowed_date - start_date 
        window_length = min(365, time_diff.days+1)
        assert 1 <= window <= window_length, (f"ValueError: Input should be an "
                                              f"integer between 1 or {window_length}.")

        end_date = start_date + pd.Timedelta(window-1, unit='d')
        test_window = pd.date_range(start=start_date, end=end_date, freq='d')

        self.test_window = test_window
        self.predict_window_validity()
        
        print(f"\nYour forecast window is days between: {test_window[0].strftime('%Y-%m-%d')} "
            f"and {test_window[-1].strftime('%Y-%m-%d')}\n")


    def get_data(self) -> Tuple[pd.DataFrame, pd.DatetimeIndex, int, int]:
        """
        Loads and preprocesses the main training dataframe and gets the forecast window.

        Returns:
            Tuple containing:
                - DataFrame with features (pd.DataFrame)
                - Forecast window (pd.DatetimeIndex)
        """
        if self.mode == "Validation":         
            file_path = self.path + "\\main_validation_dataframe.csv" 
        else:
            file_path = self.path + "\\main_testing_dataframe.csv"
        if self.feature_selection == "spd_cubed_div_temp": 
            df = self.speed_cubed_div_temp(self.make_yymmdd_format(pd.read_csv(file_path)))
        elif self.feature_selection == "speed":
            df = self.make_yymmdd_format(pd.read_csv(file_path))
        else:
            df = self.speed_cubed(self.make_yymmdd_format(pd.read_csv(file_path)))
        return df


    def run(self, display: bool = True) -> None:
        """
        Main function to load data, validate input, run kNN forecasting, and compute evaluation metrics.
        If one combination of (pca_comp, n_nbr) is given, it updates that model's prediction, scores, 
        and the plot for that combination.
        If multiple combinations are given, it runs a manual grid search and updates all predictions, 
        CV scores, and the plots for the best combinations in terms of MAPE and R2.

        Args:
            display (bool): Boolen value, determines if plots should be displayed automatically.
        """
        df = self.get_data()
        test_window  = self.test_window
        pca_comps , n_nbrs = self.pca_comps, self.n_nbrs

        true_agg = self.true_aggregate_per_day(df, test_window)
        true_aggregates = [float(x) for x in true_agg.values()]
        self.true_aggregates = true_aggregates

        print("Cross validation running on your prediction window...\n")
        if (len(pca_comps) == 1) and (len(n_nbrs) == 1):
            pca_comp = pca_comps[0]
            n_nbr = n_nbrs[0]
            preds = self.knn_using_particular_hyperparams_and_test_window(df, 
                                                    pca_comp, n_nbr, test_window)
        
            if self.ci:
                print("Running with iterations... This may take longer time.")
                alpha = 0.95
                lower, upper = 100 * (1 - alpha) / 2, 100 * (1 + alpha) / 2
                no_days = len(test_window)
                no_iter = self.no_iters

                preds = np.array(preds)
                mapes_per_iter= [mean_absolute_percentage_error(y_pred = preds[:,i], 
                                            y_true = true_aggregates) for i in range(no_iter)]
                r2s_per_iter = [r2_score(y_pred = preds[:,i], 
                                            y_true = true_aggregates) for i in range(no_iter)]
                maes_per_iter = [mean_absolute_error(y_pred = preds[:,i], 
                                            y_true = true_aggregates) for i in range(no_iter)]

                mape_mean = np.mean(mapes_per_iter)
                r2_mean = np.mean(r2s_per_iter)
                mae_mean = np.mean(maes_per_iter) 

                self.pred_conf_intervals = [(np.percentile(preds, lower, axis=1)[i], 
                                             np.percentile(preds, upper, axis=1)[i]) for i in range(no_days)]
                self.mape_conf_intervals = (np.percentile(mapes_per_iter, lower), 
                                            np.percentile(mapes_per_iter, upper))
                self.r2_conf_intervals = (np.percentile(r2s_per_iter, lower), 
                                            np.percentile(r2s_per_iter, upper))
                self.mae_conf_intervals = (np.percentile(maes_per_iter, lower), 
                                            np.percentile(maes_per_iter, upper))
                print(f"Error score: mean MAPE = {mape_mean:.3f}, mean R2= {r2_mean:.3f}, " 
                      f"mean MAE = {mae_mean}.")
                print(f"Confidence interval for MAPE: {self.mape_conf_intervals}.")
                print(f"Confidence interval for R2: {self.r2_conf_intervals}.")
                print(f"Confidence interval for MAE: {self.mae_conf_intervals}.")
                self.preds, self.mape, self.r2, self.mae = preds, mape_mean, r2_mean, mae_mean
                self.figure = self.plot(preds, true_agg, mape_mean, r2_mean, mae_mean, pca_comp, n_nbr)
            else:
                mape= mean_absolute_percentage_error(y_pred = preds, y_true = true_aggregates)
                r2 = r2_score(y_pred = preds, y_true = true_aggregates)
                mae = mean_absolute_error(y_pred=preds, y_true= true_aggregates)

                self.preds, self.mape, self.r2, self.mae = preds, mape, r2, mae
                self.figure = self.plot(preds, true_agg, mape, r2, mae, pca_comp, n_nbr)
            if display:
                self.figure.show()

        else:
            hyperparam_comb_index = {}
            hyperparam_comb_preds = []
            cv_mapes = []
            cv_r2s = []
            cv_maes = []
    
            for i in range(len(pca_comps)):
                for j in range(len(n_nbrs)):
                    hyperparam_comb_index[len(n_nbrs)*i + j] = (pca_comps[i], n_nbrs[j])
                    try:
                        preds = self.knn_using_particular_hyperparams_and_test_window(df, 
                                                            pca_comps[i], n_nbrs[j], test_window)
                        mape= mean_absolute_percentage_error(y_pred = preds, y_true = true_aggregates)
                        r2 = r2_score(y_pred = preds, y_true = true_aggregates)
                        mae = mean_absolute_error(y_pred=preds, y_true= true_aggregates)
                        cv_mapes.append(mape)
                        cv_r2s.append(r2)
                        cv_maes.append(mae)
                        
                        hyperparam_comb_preds.append(preds)

                        print(f"Prediction done with: PCA comps = {pca_comps[i]}, knn nbr = {n_nbrs[j]}. "
                            f"Error scores: cv_mape = {mape:.3f}, cv_R2 = {r2:.3f}, MAE = {mae:.1f}.")
                    except:
                        print(f"Error in hyperparameters combination: {(pca_comps[i], n_nbrs[j])}")
            self.hyperparam_comb_index = hyperparam_comb_index
            self.hyperparam_comb_preds = hyperparam_comb_preds  
            self.cv_mapes, self.cv_r2s, self.cv_maes = cv_mapes, cv_r2s, cv_maes
            self.find_best_hyperparams_combination()
            print("\nPloting the best predicted values for the best parameter combination...")

            best_MAPE_idx, best_r2_idx, best_mae_idx = (self.best_MAPE_idx, 
                                                        self.best_r2_idx,
                                                        self.best_mae_idx)

            self.fig_best_MAPE = self.plot(hyperparam_comb_preds[best_MAPE_idx], 
                                    true_agg, cv_mapes[best_MAPE_idx], 
                                    cv_r2s[best_MAPE_idx], cv_maes[best_MAPE_idx], 
                                    hyperparam_comb_index[best_MAPE_idx][0], 
                                    hyperparam_comb_index[best_MAPE_idx][1])
            self.fig_best_r2 = self.plot(hyperparam_comb_preds[best_r2_idx], 
                                    true_agg, cv_mapes[best_r2_idx], 
                                    cv_r2s[best_r2_idx],cv_maes[best_r2_idx], 
                                    hyperparam_comb_index[best_r2_idx][0], 
                                    hyperparam_comb_index[best_r2_idx][1])
            self.fig_best_mae = self.plot(hyperparam_comb_preds[best_mae_idx], 
                                    true_agg, cv_mapes[best_mae_idx], 
                                    cv_r2s[best_mae_idx], cv_maes[best_mae_idx],  
                                    hyperparam_comb_index[best_mae_idx][0], 
                                    hyperparam_comb_index[best_mae_idx][1])
            if display:
                print("Best in terms of MAPE:")
                self.fig_best_MAPE.show()
                print("Best in terms of R2:")               
                self.fig_best_r2.show()
                print("Best in terms of MAE:")
                self.fig_best_mae.show()




    def find_best_hyperparams_combination(self) -> None:
        """
        Finds and prints the best hyperparameter combinations based on MAPE and R² scores.

        Args:
            cv_mapes (List[float]): List of MAPE scores for each hyperparameter combination.
            cv_r2s (List[float]): List of R² scores for each hyperparameter combination.
            hyperparam_comb_index (Dict[int, Tuple[int, int]]): Index-to-hyperparameter mapping.

        Returns:
            Tuple[int, int]: Indices of the best hyperparameter combination based on MAPE and R².
        """
        cv_mapes, cv_r2s, cv_maes = self.cv_mapes, self.cv_r2s, self.cv_maes 
        hyperparam_comb_index = self.hyperparam_comb_index
        best_MAPE_idx = cv_mapes.index(min(cv_mapes))
        best_r2_idx = cv_r2s.index(max(cv_r2s))
        best_mae_idx = cv_maes.index(min(cv_maes))

        print(f"\n\nBest combination in terms of MAPE: pca = {hyperparam_comb_index[best_MAPE_idx][0]},"
            f" nbr = {hyperparam_comb_index[best_MAPE_idx][1]}, MAPE = {cv_mapes[best_MAPE_idx]:.3f},"
            f" R2 = {cv_r2s[best_MAPE_idx]:.3f}, MAE = {cv_maes[best_MAPE_idx]:.1f}.")
        print(f"Best combination in terms of R2 score: pca = {hyperparam_comb_index[best_r2_idx][0]},"
            f" nbr = {hyperparam_comb_index[best_r2_idx][1]}, MAPE = {cv_mapes[best_r2_idx]:.3f},"
            f" R2 = {cv_r2s[best_r2_idx]:.3f}, MAE = {cv_maes[best_r2_idx]:.1f}.")
        print(f"Best combination in terms of MAE score: pca = {hyperparam_comb_index[best_mae_idx][0]},"
            f" nbr = {hyperparam_comb_index[best_mae_idx][1]}, MAPE = {cv_mapes[best_mae_idx]:.3f},"
            f" R2 = {cv_r2s[best_mae_idx]:.3f}, MAE = {cv_maes[best_mae_idx]:.1f}.")
        self.best_MAPE_idx = best_MAPE_idx
        self.best_r2_idx = best_r2_idx
        self.best_mae_idx = best_mae_idx
        



