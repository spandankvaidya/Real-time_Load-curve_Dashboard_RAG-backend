# app/dash_app.py
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from model import run_prediction_for_date

def create_dash_app():
    dash_app = Dash(__name__, requests_pathname_prefix='/dashboard/')
    dash_app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='prediction-data-store'),
        html.H3("🔴 Real-time Load Curve", style={"textAlign": "center"}),
        html.H4(id='selected-date-display'),
        dcc.Graph(id='live-graph', style={"height": "80vh"}),
        dcc.Interval(id='interval', interval=1000, n_intervals=0)
    ])

    @dash_app.callback(
        Output('prediction-data-store', 'data'),
        Output('selected-date-display', 'children'),
        Input('url', 'pathname')
    )
    def load_data_from_url(pathname):
        try:
            date_str = pathname.strip().split('/')[-1]
            data = run_prediction_for_date(date_str)
            if data:
                return data, f"📅 Forecast for: {date_str}"
        except Exception as e:
            print(f"Error processing Dash pathname {pathname}: {e}")
        return {}, "❌ Error: Could not load data for the selected date."

    @dash_app.callback(
        Output('live-graph', 'figure'),
        Input('interval', 'n_intervals'),
        State('prediction-data-store', 'data')
    )
    def update_graph(n, data):
        if not data:
            return {'data': [], 'layout': go.Layout(title='Waiting for data...')}
        time_ticks = data.get('timestamps', [])
        predicted_values = data.get('predicted_values', [])
        actual_values = data.get('actual_values', [])
        return {
            'data': [
                go.Scatter(x=time_ticks[:n], y=predicted_values[:n], name='Predicted', line=dict(color='blue')),
                go.Scatter(x=time_ticks[:n], y=actual_values[:n], name='Actual', line=dict(color='red'))
            ],
            'layout': go.Layout(
                xaxis={'title': 'Time'},
                yaxis={'title': 'Power Consumption'},
                height=600,
                transition={'duration': 500}
            )
        }
    return dash_app
