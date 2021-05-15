import plotly.graph_objects as go
from optionsPy import combine
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

RANGE = 200

app = dash.Dash(__name__)

df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')

available_indicators = df['Indicator Name'].unique()

app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='action',
                options=[{'label': "Buy", 'value': "Buy"}, {'label': "Sell", 'value': "Sell"}],
                style={'width': '48%', 'display': 'inline-block'})]),

        html.Div([
            dcc.Dropdown(
                id='type',
                options=[{'label': "Call", 'value': "Call"}, {'label': "Put", 'value': "Put"}],
                style={'width': '48%', 'float': 'right', 'display': 'inline-block'})]),
        dcc.Input(id="price", value="")
        dcc.Graph(id='graph')])])

@app.callback(
    Output("graph", "figure"),
    [Input("df", "value")]
)

def update_figure(action, type, p):
    if action == "Buy":
        df = combine.long_option_payoff()
    if action == "Sell":
        df = combine.short_option_payoff()
    return {
        'data': df,
        'layout': [dict(
            xaxis={
                'title': "Future Spot Price",
                "type": "linear",
                "range":[0, RANGE]
            },
            yaxis={
                'title': "Return"
            }
        )]

    }


if __name__ == '__main__':
    app.run_server(debug=True)

