import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import pandas as pd
from datetime import date
from knn_model.hyperparams_tuning_and_model_selection.kNN_script import kNN_Cross_Validation
from knn_model.real_time_forecast.kNN_real_time_script import kNN_forecast
from arima_model.models.timeseries import ARIMAxPredictions

app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Power Prediction Dashboard"),
    html.Label("Select a Model"),
    dcc.Dropdown(options = ['kNN (Real-time forecasting)', 'kNN (Validation/Testing)','ARIMAx'], 
                 value = None , id='dropdown'),
    html.Label("Select Prediction Start Date "),
    dcc.DatePickerSingle(
        id='start-date-picker',
        min_date_allowed= None,
        max_date_allowed= None,
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
    Output('start-date-picker', 'min_date_allowed'),
    Output('start-date-picker', 'max_date_allowed'),
    Output('start-date-picker', 'date'),
    Output('forecast-window', 'value'),
    Output('forecast-graph', 'figure'),
    Output('output-date-range', 'children'),
    Output('output-error-scores', 'children'),
    Input('dropdown', 'value'),
    Input('submit-button', 'n_clicks'),
    State('start-date-picker', 'date'),
    State('forecast-window', 'value')
)
def update_all(model_choice, n_clicks, start_date, window):
    triggered = callback_context.triggered_id

    if triggered == 'dropdown':
        # User changed model: reset everything
        if model_choice in ['kNN (Validation/Testing)','ARIMAx']:
            min_date = pd.to_datetime('2019-03-03')
            max_date = pd.to_datetime('2023-12-31')
        else:
            # Nothing selected
            min_date = pd.to_datetime('2024-08-20')
            max_date = pd.Timestamp(f'{str(date.today())}')

        return min_date, max_date, None, None, go.Figure(), "", ""

    elif triggered == 'submit-button':
        # User clicked submit → run forecast
        if not start_date or not window:
            msg = "Please provide both start date and prediction window."
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, go.Figure(), msg, ""

        end_date = pd.to_datetime(start_date) + pd.Timedelta(days=window-1)

        if model_choice == 'kNN (Validation/Testing)':
            knn = kNN_Cross_Validation(with_confidence_interval=False, mode="Testing")
            error_msg = knn.manual_input(start_date, window)
            if error_msg:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, go.Figure(), error_msg, ""
            knn.run(display=False)
            fig = knn.figure
            cmape, cmae, cr2 = knn.mape, knn.mae, knn.r2
            output_scores = f"kNN CV error scores: MAPE: {cmape:.3f}; MAE: {cmae:.3f}, R²: {cr2:.3f}."
        elif model_choice == 'ARIMAx':
            arima = ARIMAxPredictions()
            arima.load_training_data(path='../1_data/final_dataframes/main_testing_dataframe.csv')
            cmape, cmae, cr2 = arima.forecast_range(start_date, end_date, plotting=False)
            fig = arima.plot_plotly(start_date, end_date)
            output_scores = f"ARIMAx CV error scores: MAPE: {cmape:.3f}; MAE: {cmae:.3f}, R²: {cr2:.3f}."
        elif model_choice == 'kNN (Real-time forecasting)':
            knn = kNN_forecast()
            error_msg = knn.set_input(start_date, window)
            if error_msg:
                return go.Figure(), error_msg
            knn.forecast(display=False)
            fig = knn.figure
            cmape, cmae, cr2 = knn.mape, knn.mae, knn.r2
            if cmape:
                ## Forecast window in the past, meaning we have true power output values to compute error
                output_scores = f"kNN CV error scores: MAPE: {cmape:.3f}; MAE: {cmae:.3f}, R²: {cr2:.3f}."
            else:
                ## True power outputs not available, not possible to calculate error
                output_scores = (f"Predicted values: {[f"{x: .1f}" for x in knn.predicted_values]}" 
                                 if window <=3 else "")
        else:
            return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                    go.Figure(), "Invalid model", "")

        return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                fig, f"Prediction window: {start_date} to {end_date.date()}", output_scores)

    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, "", ""

if __name__ == '__main__':
    app.run_server(debug=True)