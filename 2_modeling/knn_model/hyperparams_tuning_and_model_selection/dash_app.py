import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from knn_model.hyperparams_tuning_and_model_selection.kNN_script import kNN_Cross_Validation

app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Power Forecast Dashboard"),
    html.Label("Select Model"),
    dcc.Dropdown(options = ['kNN','ARIMA'], value = None , id='dropdown'),
    html.Label("Select Forecast Start Date"),
    dcc.DatePickerSingle(
        id='start-date-picker',
        min_date_allowed=pd.to_datetime('2019-03-03'),
        max_date_allowed=pd.to_datetime('2022-12-31'),
        date=None
    ),

    html.Br(),

    html.Label("Forecast Window (in days)"),
    dcc.Input(id='forecast-window', type='number', min=1, max=365, value= None),

    html.Br(), html.Br(),

    html.Button("Submit", id='submit-button', n_clicks=0),

    html.Div(id='output-date-range'),

    dcc.Graph(id='forecast-graph')
])

@app.callback(
    Output('forecast-graph', 'figure'),
    Output('output-date-range', 'children'),
    Input('submit-button', 'n_clicks'),
    State('dropdown', 'value'),
    State('start-date-picker', 'date'),
    State('forecast-window', 'value')
)
def update_forecast(n_clicks, choice, start_date, window):
    if not start_date or not window:
        return go.Figure(), "Please provide both start date and forecast window."
    if choice == 'kNN':
        knn = kNN_Cross_Validation(days_to_train_on=90)
        # Pass input
        error_msg = knn.manual_input(start_date, window)

        if error_msg:
            return go.Figure(), error_msg  # Empty plot + error message

        # Valid input → make forecast
        knn.run(display=False)
        fig = knn.figure
    else:
        pass
    end_date = pd.to_datetime(start_date) + pd.Timedelta(days=window-1)
    return fig, f"Forecast window is from {start_date} to {end_date.date()}"


if __name__ == '__main__':
    app.run_server(debug=True)