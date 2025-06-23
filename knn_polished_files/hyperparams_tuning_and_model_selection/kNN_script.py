import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from typing import Tuple, List, Union, Dict



def list_of_days(time_window: pd.DatetimeIndex) -> List[str]:
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


def plot(preds: List[float], true: Dict[str, float], mape: float, r2: float, pca_comp: int, n_nbr: int) -> None:
    """
    Plots true vs predicted values for a prediction time window and annotates MAPE and R².

    Args:
        preds (List[float]): List of predicted values.
        true (Dict[str, float]): Dictionary with date as key and the true aggregate power for the day as value.
        mape (float): Mean Absolute Percentage Error.
        r2 (float): R² score.
        pca_comp (int): PCA component used for the model.
        n_nbr (int): Number of kNN neighbours used for the model. 
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    days = [str(x) for x in true.keys()]
    true_vals = [float(x) for x in true.values()]

    ax.plot(days, true_vals, label = 'True')
    ax.plot(days, preds, label = "Predicted")
    ax.legend(loc='lower right')
    ax.tick_params(axis='x', rotation = 45)
    ax.set_xticks([days[i] for i in range(1,len(days),len(days)// 12)])
    ax.set_xlabel('Date')
    ax.set_ylabel('Total power per day (in MW)')
    ax.set_title("Plot of total power per day showing true and predicted.")

    # Annotate MAPE and R² on plot
    annotation_text = f"PCA: {pca_comp} | kNN nbr: {n_nbr} | MAPE: {mape:.3f} | R²: {r2:.3f}"
    ax.text(0.38, 0.95, annotation_text, transform=ax.transAxes,
        fontsize=12,horizontalalignment= 'left', verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()

    plt.show()



def make_yymmdd_format(df: pd.DataFrame) -> pd.DataFrame:
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

def speed_cubed_div_temp(df: pd.DataFrame) -> pd.DataFrame:
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


def true_aggregate_per_day(df: pd.DataFrame, time_window: pd.DatetimeIndex) -> Dict[str, float]:
    """
    Computes (true) total wind power per day over a given time window.

    Args:
        df (pd.DataFrame): DataFrame with 'YearMonthDay' and 'Wind' columns.
        time_window (pd.DatetimeIndex): Range of dates to compute aggregates for.

    Returns:
        Dict[str, float]: A dictionary mapping 'yy-mm-dd' strings to total wind production values.
    """
    try:
        assert type(time_window) ==  pd.core.indexes.datetimes.DatetimeIndex, ("TypeError: "
                                "time_window must be of type pandas.core.indexes.datetimes.DatetimeIndex.")
        day_list = list_of_days(time_window)
        df_test = df[df.YearMonthDay.isin(day_list)]
        aggregates = {}
        for day in day_list:
            df_day = df_test[df_test['YearMonthDay'] == day]
            aggregates[day] = sum(np.array(df_day.Wind))
        return aggregates 
    except:
        print("Incorrect dataframe passed to true_aggregate_per_day.")


def extract_train_test_data(
        df: pd.DataFrame, train_window: pd.DatetimeIndex, test_window: pd.DatetimeIndex
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        train_days = list_of_days(train_window)
        test_days = list_of_days(test_window)
        train_df = df[df['YearMonthDay'].isin(train_days)]
        test_df = df[df['YearMonthDay'].isin(test_days)]
        return train_df, test_df
    except:
        print("Incorrect dataframe passed to extract_train_test_data.")

def predict_window_validity(test_window: pd.DatetimeIndex) -> None:
    """
    Validates the format and range of the forecast window.

    Args:
        test_window (pd.DatetimeIndex): Forecast window to validate.

    Raises:
        AssertionError: If the input is not a DatetimeIndex or out of allowed range.
    """
    assert isinstance(test_window, pd.DatetimeIndex), ("TypeError: test_window must be of type "
            "pandas.core.indexes.datetimes.DatetimeIndex")
    test_window = test_window.sort_values()
    assert (pd.Timestamp(year=2020, month=1, day=1) <= test_window[0] 
            and test_window[-1] <= pd.Timestamp(year=2023, month=12, day=31)), ("ValueError: test_window must be "
            "between 2020-01-01 to 2023-12-31")
  

def kNN_on_particular_train_test_splits(
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


def knn_using_particular_hyperparams_and_test_window(
    df: pd.DataFrame, pca_comp: int, n_nbr: int, test_window: pd.DatetimeIndex
) -> Tuple[List[float], List[float]]:
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
        List[float]: Predicted wind outputs per day.
    """
    preds = []

    for test_day in test_window:
        test_day = pd.date_range(test_day,test_day,freq='d') ## pd.Timestamp to pd.DatetimeIndex for function call
        
        train_window = get_train_window(test_day)   
        df_train, df_test = extract_train_test_data(df, train_window, test_day)

        pred = kNN_on_particular_train_test_splits(df_train, df_test, pca_comp, n_nbr)
        preds.append(pred[0])

    return preds

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

def get_forecast_window() -> pd.DatetimeIndex:
    """
    Interactively prompts the user to input a forecast window.

    Returns:
        pd.DatetimeIndex: A pandas date range between selected start and end date.
    """
    earliest_allowed_date = '2019-03-02'
    latest_allowed_date = '2023-12-31'
    print(f"\nPlease enter forecast start date in the format YYYY-MM-DD." 
          f"The date should be in between {earliest_allowed_date} to {latest_allowed_date}.")
    date= input("Input the start date now: ")
    try:
        year, month, day = int(date[:4]), int(date[5:7]), int(date[8:])
        assert 2019 <= year <=2023, "ValueError: Input a year between 2019 to 2023."
        start_date = pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        raise AssertionError(f"Invalid date: {year}-{month}-{day}")
    print(f"\nYou have entered: {start_date.strftime('%Y-%m-%d')}.")

    time_diff = pd.Timestamp(year=2022,month=12,day=31) - start_date  ### need to change this to 2023 later
    window_length = min(365, time_diff.days+1)
    print(f"\nPlease enter forecast window (in days). It should be an integer between 1 or {window_length}.")
    
    window = int(input("Input forecast window: "))
    assert 1 <= window <= window_length, f"ValueError: Input should be an integer between 1 or {window_length}."

    end_date = start_date + pd.Timedelta(window-1, unit='d')
    predict_window = pd.date_range(start=start_date, end=end_date, freq='d')
    
    print(f"\nYour forecast window is days between: {predict_window[0].strftime('%Y-%m-%d')} "
          f"and {predict_window[-1].strftime('%Y-%m-%d')}\n")
    return predict_window


def get_data() -> Tuple[pd.DataFrame, pd.DatetimeIndex, int, int]:
    """
    Loads and preprocesses the main training dataframe and gets the forecast window.

    Returns:
        Tuple containing:
            - DataFrame with features (pd.DataFrame)
            - Forecast window (pd.DatetimeIndex)
    """          
    file_path = os.path.join(os.path.dirname(__file__), "../../data/final_dataframes/main_training_dataframe.csv")
                                    ## need to include the main dataframe instead of the training set
    df = speed_cubed_div_temp(make_yymmdd_format(pd.read_csv(file_path)))
    predict_window = get_forecast_window()
    return df, predict_window


def knn_cv_main(pca_comps: List[int] = [35], n_nbrs: List[int] = [5]) -> Union[
    Tuple[List[float], List[float], float, float],
    Tuple[Dict[int, Tuple[int, int]], List[List[float]], List[float], List[float], List[float]]
]:
    """
    Main function to load data, validate input, run kNN forecasting, and compute evaluation metrics.
    If one combination of (pca_comp, n_nbr) is given, it returns that model's prediction and scores.
    If multiple combinations are given, it runs a manual grid search and returns all predictions and CV scores.

    Args:
        pca_comps (List[int], optional): List of PCA component counts to test. Default is [35].
        n_nbrs (List[int], optional): List of kNN neighbor values to test. Default is [5].

    Returns:
        Union:
            - Tuple of predicted values, true values, MAPE, R² score (if only one hyperparam combo)
            - Tuple of all combinations, all predictions, true values, list of MAPE scores, list of R² scores
    """
    df, predict_window = get_data()
    predict_window_validity(predict_window)

    true_agg = true_aggregate_per_day(df, predict_window)
    true_aggregates = [float(x) for x in true_agg.values()]

    print("Cross validation running on your prediction window...\n")
    if (len(pca_comps) == 1) and (len(n_nbrs) == 1):
        pca_comp = pca_comps[0]
        n_nbr = n_nbrs[0]
        preds = knn_using_particular_hyperparams_and_test_window(df, 
                                                pca_comp, n_nbr, predict_window)

        mape= mean_absolute_percentage_error(y_pred = preds, y_true = true_aggregates)
        r2 = r2_score(y_pred = preds, y_true = true_aggregates)

        plot(preds, true_agg, mape, r2, pca_comp, n_nbr)
        return  preds, true_aggregates, mape, r2

    else:
        assert ((len(pca_comps) <= 4) and (len(n_nbrs) <=4)), ("\nPlease provide the lists of pca components"
        " and kNN neighbours of length less than 3 to reduce run-time.")

        hyperparam_comb_index = {}
        hyperparam_comb_preds = []
        cv_mapes = []
        cv_r2s = []
 
        for i in range(len(pca_comps)):
            for j in range(len(n_nbrs)):
                hyperparam_comb_index[len(n_nbrs)*i + j] = (pca_comps[i], n_nbrs[j])
                try:
                    preds = knn_using_particular_hyperparams_and_test_window(df, 
                                                        pca_comps[i], n_nbrs[j], predict_window)
                    mape= mean_absolute_percentage_error(y_pred = preds, y_true = true_aggregates)
                    r2 = r2_score(y_pred = preds, y_true = true_aggregates)
                    cv_mapes.append(mape)
                    cv_r2s.append(r2)
                    hyperparam_comb_preds.append(preds)

                    print(f"Prediction done with: PCA comps = {pca_comps[i]}, knn nbr = {n_nbrs[j]}. "
                          f"Error scores: cv_mape = {mape:.3f}, cv_R2 = {r2:.3f}.")
                except:
                    print(f"Error in hyperparameters combination: {(pca_comps[i], n_nbrs[j])}")
        best_MAPE_idx, best_r2_idx = find_best_hyperparams_combination(cv_mapes, 
                                                        cv_r2s, hyperparam_comb_index)
        print("\nPloting the best predicted values for the best parameter combination...")
        plot(hyperparam_comb_preds[best_MAPE_idx], true_agg, cv_mapes[best_MAPE_idx], 
             cv_r2s[best_MAPE_idx], hyperparam_comb_index[best_MAPE_idx][0], hyperparam_comb_index[best_MAPE_idx][1])
        plot(hyperparam_comb_preds[best_r2_idx], true_agg, cv_mapes[best_r2_idx], 
             cv_r2s[best_r2_idx], hyperparam_comb_index[best_r2_idx][0], hyperparam_comb_index[best_r2_idx][1])
        return hyperparam_comb_index, hyperparam_comb_preds,  true_aggregates, cv_mapes, cv_r2s



def find_best_hyperparams_combination(
    cv_mapes: List[float],
    cv_r2s: List[float],
    hyperparam_comb_index: Dict[int, Tuple[int, int]]
) -> Tuple[int, int]:
    """
    Finds and prints the best hyperparameter combinations based on MAPE and R² scores.

    Args:
        cv_mapes (List[float]): List of MAPE scores for each hyperparameter combination.
        cv_r2s (List[float]): List of R² scores for each hyperparameter combination.
        hyperparam_comb_index (Dict[int, Tuple[int, int]]): Index-to-hyperparameter mapping.

    Returns:
        Tuple[int, int]: Indices of the best hyperparameter combination based on MAPE and R².
    """
    best_MAPE_idx = cv_mapes.index(min(cv_mapes))
    best_r2_idx = cv_r2s.index(max(cv_r2s))

    print(f"\n\nBest combination in terms of MAPE: pca = {hyperparam_comb_index[best_MAPE_idx][0]},"
          f" nbr = {hyperparam_comb_index[best_MAPE_idx][1]}, MAPE = {cv_mapes[best_MAPE_idx]:.3f},"
          f" R2 = {cv_r2s[best_MAPE_idx]:.3f}")
    print(f"Best combination in terms of R2 score: pca = {hyperparam_comb_index[best_r2_idx][0]},"
          f" nbr = {hyperparam_comb_index[best_r2_idx][1]}, MAPE = {cv_mapes[best_r2_idx]:.3f},"
          f" R2 = {cv_r2s[best_r2_idx]:.3f}")
    return best_MAPE_idx, best_r2_idx
    



