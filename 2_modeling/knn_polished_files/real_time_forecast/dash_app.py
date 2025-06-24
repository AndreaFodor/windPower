import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from kNN_real_time_script import kNN_forecast

knn = kNN_forecast()

app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Power Forecast Dashboard"),

    html.Label("Select Forecast Start Date"),
    dcc.DatePickerSingle(
        id='start-date-picker',
        min_date_allowed=pd.to_datetime('2024-08-20'),
        max_date_allowed=pd.to_datetime('2025-06-24'),
        date='2025-04-02'
    ),

    html.Br(),

    html.Label("Forecast Window (in days)"),
    dcc.Input(id='forecast-window', type='number', min=1, max=3, value=2),

    html.Br(), html.Br(),

    html.Button("Submit", id='submit-button', n_clicks=0),

    html.Div(id='output-date-range'),

    dcc.Graph(id='forecast-graph')
])

@app.callback(
    Output('forecast-graph', 'figure'),
    Output('output-date-range', 'children'),
    Input('submit-button', 'n_clicks'),
    State('start-date-picker', 'date'),
    State('forecast-window', 'value')
)
def update_forecast(n_clicks, start_date, window):
    if not start_date or not window:
        return go.Figure(), "Please provide both start date and forecast window."

    # Pass input
    error_msg = knn.set_input(start_date, window)

    if error_msg:
        return go.Figure(), error_msg  # Empty plot + error message

    # Valid input → make forecast
    fig = knn.forecast()
    end_date = pd.to_datetime(start_date) + pd.Timedelta(days=window-1)
    return fig, f"Forecast window is from {start_date} to {end_date.date()}"


if __name__ == '__main__':
    app.run_server(debug=True)