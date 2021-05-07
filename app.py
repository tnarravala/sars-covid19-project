import os


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_daq as daq

import pandas as pd


cases = pd.read_csv("indian_cases_confirmed_cases.csv")
deaths = pd.read_csv("indian_cases_confirmed_deaths.csv")
state_dic = {'ap':'Andhra Pradesh',
             'dl':'Delhi',
             'mp':'Madhya Pradesh',
             'kl':'Kerala'}
total_cases = cases.set_index('state')
total_state_cases = total_cases.iloc[:,-1:]
total_cases = total_state_cases.sum()

total_deaths = deaths.set_index('state')
total_state_deaths = total_deaths.iloc[:,-1:]
total_deaths= total_state_deaths.sum()
def plot_cases(state,ca):
    if ca == False:
        st = cases.set_index('state')
        col1 = st['2020-01-30']
        st = st.diff(axis = 1)
        st['2020-01-30'] = col1
        st = st.reset_index()
        st = (st[st['state'] == state].T)
    else:
        st = (cases[cases['state'] == state].T)

    st = st[1:].reset_index()
    st.columns = ['date','cases']
    st['date'] = pd.to_datetime(st['date'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st['date'],y=st['cases'],mode= 'markers',name=f'{state_dic[state]}'))

    fig.update_layout(
    autosize=True,
    title = f"Cases in {state_dic[state]}",
    margin = dict(l=40, r=40, t=40, b=40 ),
    width=500,
    height=400,
    yaxis = dict(
       #range = [0,100] ,
       #rangemode="tozero",
        autorange=True,
        title_text='Cases',
        titlefont=dict(size=10),
    ),
    xaxis=dict(
        title_text = "date",
        autorange=True,
        range=["2020-01-30 18:36:37.3129", "2021-05-02 05:23:22.6871"],
        rangeslider=dict(
            autorange=True,
            range=["2020-01-30 18:36:37.3129", "2021-05-02 05:23:22.6871"]
        ),
        type="date"
    ),
    )

    return fig

def plot_deaths(state,ca):
    if ca == False:
        st = deaths.set_index('state')
        col1 = st['2020-01-30']
        st = st.diff(axis = 1)
        st['2020-01-30'] = col1
        st = st.reset_index()
        st = (st[st['state'] == state].T)
    else:
        st = (deaths[deaths['state'] == state].T)
    st = st[1:].reset_index()
    st.columns = ['date','deaths']
    st['date'] = pd.to_datetime(st['date'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st['date'],y=st['deaths'],mode= 'markers',name=f'{state_dic[state]}'))

    fig.update_layout(
    autosize=True,
    title = f"Deaths in {state_dic[state]}",

    margin = dict(l=40, r=40, t=40, b=40 ),
    width=500,
    height=400,
    yaxis = dict(
         #range = [0,100] ,
         #rangemode="tozero",
        autorange=True,
        title_text='deaths',
        titlefont=dict(size=10),
    ),
    xaxis=dict(
        title_text = "date",
        autorange=True,
        range=["2020-01-30 18:36:37.3129", "2021-05-02 05:23:22.6871"],
        rangeslider=dict(
            autorange=True,
            range=["2020-01-30 18:36:37.3129", "2021-05-02 05:23:22.6871"]
        ),
        type="date"
    ),
    )

    return fig
def plot_total_cases(ca):
    if ca == False:
        st = cases.set_index('state')
        col1 = st['2020-01-30']
        st = st.diff(axis = 1)
        st['2020-01-30'] = col1
        st = st.reset_index()
    else:
        st = cases
    ind = st.sum(axis =0)[1:]
    ind = ind.to_frame()
    ind = ind.reset_index()
    ind.columns = ['date','sum']
    ind['date'] = pd.to_datetime(ind['date'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ind['date'],y=ind['sum'],mode= 'markers'))

    fig.update_layout(
    autosize=True,
    title = "Cases in India",
    margin = dict(l=40, r=40, t=40, b=40 ),
    width=500,
    height=400,
    yaxis = dict(
       #range = [0,100] ,
       #rangemode="tozero",
        autorange=True,
        title_text='cases',
        titlefont=dict(size=10),
    ),
    xaxis=dict(
        title_text = "date",
        autorange=True,
        range=["2020-01-30 18:36:37.3129", "2021-05-02 05:23:22.6871"],
        rangeslider=dict(
            autorange=True,
            range=["2020-01-30 18:36:37.3129", "2021-05-02 05:23:22.6871"]
        ),
        type="date"
    ),
    )

    return fig

def plot_total_deaths(ca):
    if ca == False:
        st = deaths.set_index('state')
        col1 = st['2020-01-30']
        st = st.diff(axis = 1)
        st['2020-01-30'] = col1
        st = st.reset_index()
    else:
        st = deaths
    ind = st.sum(axis =0)[1:]
    ind = ind.to_frame()
    ind = ind.reset_index()
    ind.columns = ['date','sum']
    ind['date'] = pd.to_datetime(ind['date'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ind['date'],y=ind['sum'],mode= 'markers'))

    fig.update_layout(
    autosize=True,
    title = "Deaths in India",
    margin = dict(l=40, r=40, t=40, b=40 ),
    width=500,
    height=400,
    yaxis = dict(
       #range = [0,100] ,
       #rangemode="tozero",
        autorange=True,
        title_text='deaths',
        titlefont=dict(size=10),
    ),
    xaxis=dict(
        title_text = "date",
        autorange=True,
        range=["2020-01-30 18:36:37.3129", "2021-05-02 05:23:22.6871"],
        rangeslider=dict(
            autorange=True,
            range=["2020-01-30 18:36:37.3129", "2021-05-02 05:23:22.6871"]
        ),
        type="date"
    ),
    )

    return fig

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

body = dbc.Container([ 
dbc.Row(
            [
            html.H1("Indian Covid-19 Tracker")
            ], align="center",justify = "center"
            ),
dbc.Row(
        [
            html.Div(
                [ html.P('Cummulative',style = {'display': 'inline-block'}),
            daq.ToggleSwitch(
                id='cum-act',
                value=True,
                style = {'display': 'inline-block'}
                        ),
                    ]
                
                )
            
            ]
        ),
  dbc.Row([
        dbc.Col([html.H3(id = "tsc", style = {'display': 'inline-block'})]),
        dbc.Col([html.H3(id = "tsd", style = {'display': 'inline-block'})])
        ,]),  
dbc.Row(
        [html.Br()]),
dbc.Row([
        dbc.Col(
                   
                        dcc.Graph(id="fig3",figure = plot_total_cases('Daily new cases'))
                         ,
                        
                ),
                dbc.Col(
                 
                           dcc.Graph(id="fig4",figure = plot_total_cases('Daily new cases'))
                       
                ),
               
      ]  ),
dbc.Row(
        [html.Br()]),
dbc.Row(
        [
    dcc.Dropdown(
        id='st',
        options=[
            {'label': 'Andhra Pradesh', 'value': 'ap'},
            {'label': 'Kerala', 'value': 'kl'},
            {'label': 'Madhya Pradesh', 'value': 'mp'},
            {'label':'Delhi','value':'dl'}
        ],
        value='dl',style = {'color':'black','width':'50%','display': 'inline-block','margin-left':'0.8%'}
    ),
            ]
        ),
      dbc.Row(
        [html.Br()]),     
    dbc.Row([
        dbc.Col([html.H3(id = "tc", style = {'display': 'inline-block'})]),
        dbc.Col([html.H3(id = "td", style = {'display': 'inline-block'})])
        ,]),                                                                               
    dbc.Row(
        [html.Br()]),
dbc.Row([
        dbc.Col(
                   
                       dcc.Graph(id='fig',figure = plot_deaths('ap','Daily new cases'))
                         ,
                        
                ),
                dbc.Col(
                 
                           dcc.Graph(id='fig2',figure = plot_cases('ap','Daily new cases'))
                       
                ),
               
      ]  )

],style={"height": "100vh"}

)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])

app.layout = html.Div([body])


app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

@app.callback(
    Output('fig', 'figure'),
    Input('st', 'value'),
    Input('cum-act','value'))
def update_figure(st,ca):
    fig = plot_cases(st,ca)
    fig.update_layout(transition_duration=500)
    return fig

@app.callback(
    Output('fig2', 'figure'),
    Input('st', 'value'),
    Input('cum-act','value'))
def update_figure2(st,ca):
    fig2 = plot_deaths(st,ca)
    fig2.update_layout(transition_duration=500)
    return fig2

@app.callback(
    Output('fig3', 'figure'),
    Input('cum-act','value'))
def update_figure3(ca):
    fig = plot_total_cases(ca)
    fig.update_layout(transition_duration=500)
    return fig

@app.callback(
    Output('fig4', 'figure'),
    Input('cum-act','value'))
def update_figure4(ca):
    fig = plot_total_deaths(ca)
    fig.update_layout(transition_duration=500)
    return fig
@app.callback(
    Output('tc','children'),
    Input('st','value')
    )
def update_output_div(st):
    return u'Total Cases in {}: {:,}'.format(state_dic[st],total_state_cases.loc[st].values[0]) 

@app.callback(
    Output('td','children'),
    Input('st','value')
    )
def update_output_div2(st):
    return u'Total Deaths in {}: {:,}'.format(state_dic[st],total_state_deaths.loc[st].values[0]) 

@app.callback(
    Output('tsc','children'),
    Input('st','value')
    )
def update_output_div3(st):
    return u'Total Cases in India: {:,}'.format(total_cases.values[0])

@app.callback(
    Output('tsd','children'),
    Input('st','value')
    )
def update_output_div4(st):
    return u'Total Deaths in India: {:,}'.format(total_deaths.values[0])

if __name__ == '__main__':
    app.run_server(debug=True)