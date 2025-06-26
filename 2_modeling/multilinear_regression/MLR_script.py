import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from typing import Tuple, List, Union, Dict, Optional

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV 
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.decomposition import PCA



class MlrCrossValidation:
    def __init__(self, lag_days: int = 6):
        self.lag_days = lag_days
        self.pipe= Pipeline([
                    ('scale', StandardScaler()),
                    # ('pca', PCA(n_components=0.95)),
                    # ('poly', PolynomialFeatures(degree=3, include_bias=False)),
                    ('ridge', RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)),
                      ])
        self.cv_results= None

        # Plotly figures for best results
        self.fig_best_MAPE: Optional[go.Figure] = None                      # For best combination MAPE
        self.fig_best_r2: Optional[go.Figure] = None                        # For best combination R²
    
    def data_prep(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Prepares the raw input DataFrame for modeling by performing:
            - Daily resampling and aggregation
            - Conversion of temperature to Kelvin
            - Lagged weather feature generation
            - Rolling averages for additional lag days
            - Cyclical features for day of year

            Assumes the DataFrame uses a datetime index and includes columns:
            - 'Wind'
            - columns containing 'wind_speed'
            - columns containing 'temp'

            Args:
                df (pd.DataFrame): Input DataFrame with a datetime index.

            Returns:
                pd.DataFrame: Cleaned and feature-engineered DataFrame.
        """
        wind_speed_data_columns = [col for col in df.columns if 'wind_speed' in col]
        temps_data_columns = [col for col in df.columns if 'temp' in col]
        weather_columns = wind_speed_data_columns + temps_data_columns

        # Initialize the new DataFrame with daily resampled data
        mlr_df = pd.DataFrame()
        mlr_df['Wind'] = df['Wind'].resample('D').sum()
        mlr_df[wind_speed_data_columns] = df[wind_speed_data_columns].resample('D').mean()
        mlr_df[temps_data_columns] = (df[temps_data_columns].resample('D').mean() + 273.15)**(-1)

        # Add lagged features
        if self.lag_days > 0:
            lag_df = mlr_df[weather_columns].shift(1)
            lag_df.columns = [f"{col}_lag1" for col in lag_df.columns]

            mlr_df = pd.concat([mlr_df, lag_df], axis=1)


        # Add rolling mean lag features all at once to avoid fragmentation
        if self.lag_days > 1:
            lag_window = range(2, self.lag_days)
            lagged_means = []

            for L in lag_window:
                rolled = mlr_df[weather_columns].shift(2).rolling(window=L).mean()
                rolled.columns = [f"{col}_lag{L}_mean" for col in weather_columns]
                lagged_means.append(rolled)

            # Concatenate all rolling mean lag features in one step
            mlr_df = pd.concat([mlr_df] + lagged_means, axis=1)

        # Create sine and cosine cyclical features for day of year
        delta = mlr_df.index - pd.to_datetime(mlr_df.index.year.astype(str) + '-01-01')
        mlr_df['doy_sin'] = np.sin(2 * np.pi * delta.days.astype(int) / 365)
        mlr_df['doy_cos'] = np.cos(2 * np.pi * delta.days.astype(int) / 365)

        mlr_df = mlr_df.copy()

        return mlr_df

    
    def data_for_module(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Reduces the full DataFrame to just the model-ready features and target.
        Cleans NaNs and infs, and splits into:
        - The full modeling DataFrame
        - The feature matrix X
        - The target series Y

        Args:
            df (pd.DataFrame): Pre-processed DataFrame.

        Returns:
            List[pd.DataFrame]: [df_model, X, Y]
        """
        df = self.data_prep(df)
        weather=[ S for S in df.columns if 'lag' in S]
        sesonal= ["doy_sin", "doy_cos"]

        feature_cols= weather + sesonal
        target_col= 'Wind'

        df_model = df[feature_cols +[target_col]]
        
        X= df_model[feature_cols]
        Y=df_model[target_col]

        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        Y = Y.loc[X.index]  

        return([df_model,X,Y])

    def fit(self, df: pd.DataFrame):
        """
        Fits the pipeline model to the given DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame with appropriate columns.
        """
        df_model, X, Y = self.data_for_module(df)
        self.pipe.fit(X,Y)

    def cross_validate_model(self, df:pd.DataFrame, k: int=5):
        """
        Performs k-fold cross-validation using the internal pipeline.
        Stores the results in self.cv_results and prints average metrics.

        Args:
            df (pd.DataFrame): Input DataFrame for training.
            k (int): Number of cross-validation folds.
        """
        df_model, X, Y = self.data_for_module(df)

        cv = KFold(n_splits= 5, shuffle= True, random_state= 1023)

        results = cross_validate(
            self.pipe,
            X,
            Y,
            cv= cv,
            scoring= {
                'mape': 'neg_mean_absolute_percentage_error',
                'rmse' : 'neg_root_mean_squared_error',
                'r2': 'r2'
            },
            return_train_score= False
        )

        print("Cross-Validation Results (5-fold):")
        print(f"Mean MAPE: {-results['test_mape'].mean():.2%}")
        print(f"Mean RMSE: {-results['test_rmse'].mean():.2f}")
        print(f"Mean R²: {results['test_r2'].mean():.3f}")
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates predictions from the fitted pipeline model.

        Args:
            df (pd.DataFrame): DataFrame to predict on.

        Returns:
            pd.Series: Predicted values aligned to the input index.
        """
        df_model, X, _ = self.data_for_module(df)
        preds = self.pipe.predict(X)
        return pd.Series(preds, index=X.index)      

    def plot_predictions(self, df: pd.DataFrame) -> go.Figure:
        """
        Creates a Plotly line chart comparing predicted vs. true values
        for the given DataFrame, including MAPE and R² in the title.

        Args:
            df (pd.DataFrame): DataFrame to evaluate.

        Returns:
            go.Figure: A Plotly figure showing predictions vs. actuals.
        """
        df_model, X, Y = self.data_for_module(df)
        preds = self.pipe.predict(X)
        mape = mean_absolute_percentage_error(Y, preds)
        r2 = r2_score(Y, preds)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X.index, y=Y, name="True", mode='lines'))
        fig.add_trace(go.Scatter(x=X.index, y=preds, name="Predicted", mode='lines'))
        fig.update_layout(title=f"Predictions (MAPE={mape:.2%}, R²={r2:.3f})")
        return fig
    

    def validate_model(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> go.Figure:
        """
        Fit model on training data, evaluate on validation data,
        print MAPE and MAE, and plot predictions vs actuals plus residuals.

        Args:
            df_train (pd.DataFrame): Training DataFrame.
            df_val (pd.DataFrame): Validation (hold-out) DataFrame.

        Returns:
            go.Figure: Combined Plotly figure with predicted vs actual and residuals.
        """

        #prep the data
        _, X_train, Y_train = self.data_for_module(df_train)
        _, X_test, Y_test = self.data_for_module(df_test)

        #fit the modle on training data
        self.pipe.fit(X_train, Y_train)

        #predict on test data
        preds = self.pipe.predict(X_test)

        #calculate metrics
        mape = mean_absolute_percentage_error(Y_test, preds)
        mae = mean_absolute_error(Y_test, preds)
        residuals= Y_test - preds

        print(f"Validation MAPE: {mape:.2%}")
        print(f"Validation MAE: {mae:.4f}")

            # Create subplot figure
            
        fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Predicted vs Actual", "Residuals"))

        # Predicted vs Actual plot
        fig.add_trace(go.Scatter(x=X_test.index, y=Y_test, mode='lines', name='Actual'), row=1, col=1)
        fig.add_trace(go.Scatter(x=X_test.index, y=preds, mode='lines', name='Predicted'), row=1, col=1)

        # Residuals plot
        fig.add_trace(go.Scatter(x=X_test.index, y=residuals, mode='markers', name='Residuals'), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)

        fig.update_layout(height=600, width=900,
                        title_text=f"Validation Results (MAPE={mape:.2%}, MAE={mae:.4f})")

        return fig