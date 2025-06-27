import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from knn_model.hyperparams_tuning_and_model_selection.kNN_script import kNN_Cross_Validation
from arima_model.models.timeseries import ARIMAxPredictions

app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Power Prediction Dashboard"),
    html.Label("Select a Model"),
    dcc.Dropdown(options = ['kNN','ARIMAx'], value = None , id='dropdown'),
    html.Label("Select Prediction Start Date "),
    dcc.DatePickerSingle(
        id='start-date-picker',
        min_date_allowed=pd.to_datetime('2019-03-03'),
        max_date_allowed=pd.to_datetime('2023-12-31'),
        date=None
    ),

    html.Br(),

    html.Label("Prediction Window (in days) "),
    dcc.Input(id='forecast-window', type='number', min=1, max=30, value= None),

    html.Br(), html.Br(),

    html.Button("Submit", id='submit-button', n_clicks=0),

    html.Div(id='output-date-range'),
    html.Div(id='output-error-scores'),

    dcc.Graph(id='forecast-graph')
])

@app.callback(
    Output('forecast-graph', 'figure'),
    Output('output-date-range', 'children'),
    Output('output-error-scores', 'children'),
    Input('submit-button', 'n_clicks'),
    State('dropdown', 'value'),
    State('start-date-picker', 'date'),
    State('forecast-window', 'value')
)
def update_forecast(n_clicks, choice, start_date, window):
    if not start_date or not window:
        msg = "Please provide both start date and prediction window " \
        "(restricted to 30 to manage runtime)."
        return go.Figure(), msg , " "
    
    end_date = pd.to_datetime(start_date) + pd.Timedelta(days=window-1)

    if choice == 'kNN':
        knn = kNN_Cross_Validation(with_confidence_interval=False, mode="Testing")
        # Pass input
        error_msg = knn.manual_input(start_date, window)

        if error_msg:
            return go.Figure(), error_msg  # Empty plot + error message

        # Valid input → make forecast
        knn.run(display=False)
        fig = knn.figure
        cmape, cmae, cr2 =knn.mape, knn.mae, knn.r2
        output_scores = f"kNN CV error scores: MAPE: {cmape:.3f}; MAE: {cmae:.3f}, R²: {cr2:.3f}."
    elif choice == 'ARIMAx':
        arima = ARIMAxPredictions()
        arima.load_training_data(path='../1_data/final_dataframes/main_testing_dataframe.csv')
        cmape, cmae, cr2 = arima.forecast_range(start_date, end_date, plotting=False)
        fig = arima.plot_plotly(start_date, end_date)
        output_scores = f"ARIMAx CV error scores: MAPE: {cmape:.3f}; MAE: {cmae:.3f}, R²: {cr2:.3f}."

    return fig, f"Prediction window is from {start_date} to {end_date.date()}.", output_scores


if __name__ == '__main__':
    app.run_server(debug=True)