import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.express as px
import pandas as pd


cases = pd.read_csv("indian_cases_confirmed_cases.csv")
deaths = pd.read_csv("indian_cases_confirmed_deaths.csv")
state_dic = {'ap':'Andhra Pradesh',
             'dl':'Delhi',
             'mp':'Madhya Pradesh',
             'kl':'Kerala',
             'up':'Uttar Pradesh',
             'mh':'Maharastra',
             'br':'Bihar',
             'wb':'West Bengal',
             'tn':'Tamil Nadu',
             'rj':'Rajesthan',
             'ka':'Karnataka',
             'gj':'Gujarat',
             'or':'Odisha',
             'tg':'Telangana',
             'jh':'Jharkhand',
             'as':'Assam',
             'pb':'Punjab',
             'ct':'Chattisgarh',
             'hr':'Haryana',
             'jk':'Jammu and Kashmir',
             'ut':'Uttarakhand',
             'hp':'Himachal Pradesh',
             'tr':'Tripura',
             'ml':'Meghalaya',
             'mn':'Manipur',
             'nl':'Nagaland',
             'ga':'Goa',
             'ar':'Arunachal Pradesh',
             'py':'Puducherry',
             'mz':'Mizoram',
             'ch':'Chandigarh',
             'sk':'Sikkim',
             'dn_dd':'Daman and Diu',
             'an':'Andaman and Nicobar',
             'ld':'Ladakh',
             'la':'Lakshdweep'}
total_cases = cases.set_index('state')
total_state_cases = total_cases.iloc[:,-1:]
total_cases = total_state_cases.sum()

total_deaths = deaths.set_index('state')
total_state_deaths = total_deaths.iloc[:,-1:]
total_deaths= total_state_deaths.sum()

date_range = ["2020-01-30 18:36:37.3129", "2021-05-07 05:23:22.6871"]

def plot_cases(state,ca):
    if ca == True:
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
    st_name = u'Cases in {}'.format(state_dic[state])
    #fig = go.Figure()
    #fig.add_trace(go.Scatter(x=st['date'],y=st['cases'],mode= 'markers',name=f'{state_dic[state]}'))
    fig = px.bar(st, x='date', y='cases')
    fig.update_layout(
    autosize=True,
    #title = st_name,
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
        range=date_range,
        rangeslider=dict(
            autorange=True,
            range=date_range
        ),
        type="date"
    ),
    )

    return fig

def plot_deaths(state,ca):
    if ca == True:
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
    #fig = go.Figure()
    #fig.add_trace(go.Scatter(x=st['date'],y=st['deaths'],mode= 'markers',name=f'{state_dic[state]}'))
    st_name = u'Deaths in {}'.format(state_dic[state])
    fig = px.bar(st, x='date', y='deaths')
    fig.update_layout(
    autosize=True,
    #title =  st_name,

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
        range=date_range,
        rangeslider=dict(
            autorange=True,
            range=date_range
        ),
        type="date"
    ),
    )

    return fig
def plot_total_cases(ca):
    if ca == True:
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
    #fig = go.Figure()
    #fig.add_trace(go.Scatter(x=ind['date'],y=ind['sum'],mode= 'markers'))
    fig = px.bar(ind, x='date', y='sum')
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
        range=date_range,
        rangeslider=dict(
            autorange=True,
            range=date_range
        ),
        type="date"
    ),
    )

    return fig

def plot_total_deaths(ca):
    if ca == True:
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
    #fig = go.Figure()
    #fig.add_trace(go.Scatter(x=ind['date'],y=ind['sum'],mode= 'markers'))
    fig = px.bar(ind, x='date', y='sum')
    fig.update_layout(
    autosize=True,
    title = "Deaths in India",
    margin = dict(l=40, r=40, t=40, b=40 ),
    width=500,
    height=400,

    #style = {'color':'green'},
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
        range=date_range,
        rangeslider=dict(
            autorange=True,
            range=date_range
        ),
        type="date"
    ),
    )

    return fig

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']





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
        dbc.Col([html.H3(id = "tsc", style = {'display': 'inline-block'}),
                 dcc.Graph(id="fig3",figure = plot_total_cases('Daily new cases'))]),
        dbc.Col([html.H3(id = "tsd", style = {'display': 'inline-block'}),
                  dcc.Graph(id="fig4",figure = plot_total_cases('Daily new cases'))])
        ,],align='center',justify = "center"),
  
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
            {'label':'Delhi','value':'dl'},
            {'label':'Uttar Pradesh','value':'up'},
            {'label':'Maharastra','value':'mh'},
            {'label':'Bihar','value':'br'},
            {'label':'West Bengal','value':'wb'},
            {'label':'Tamil Nadu','value':'tn'},
            {'label':'Rajesthan','value':'rj'},
            {'label':'Karnataka','value':'ka'},
            {'label':'Gujarat','value':'gj'},
            {'label':'Odisha','value':'or'},
            {'label':'Telangana','value':'tg'},
            {'label':'Jharkhand','value':'jh'},
            {'label':'Assam','value':'as'},
            {'label':'Punjab','value':'pb'},
            {'label':'Chattisgarh','value':'ct'},
            {'label':'Haryana','value':'hr'},
            {'label':'Jammu and Kashmir','value':'jk'},
            {'label':'Uttarakhand','value':'ut'},
            {'label':'Himachal Pradesh','value':'hp'},
            {'label':'Tripura','value':'tr'},
            {'label':'Meghalaya','value':'ml'},
            {'label':'Manipur','value':'mn'},
            {'label':'Nagaland','value':'nl'},
            {'label':'Goa','value':'ga'},
            {'label':'Arunachal Pradesh','value':'ar'},
            {'label':'Puducherry','value':'py'},
            {'label':'Mizoram','value':'mz'},
            {'label':'Chandigarh','value':'ch'},
            {'label':'Sikkim','value':'sk'},
            {'label':'Daman and Diu','value':'dn_dd'},
            {'label':'Andaman and Nicobar','value':'an'},
            {'label':'Ladakh','value':'ld'},
            {'label':'Lakshdweep','value':'la'},
            
        ],
        value='dl',style = {'color':'black','width':'50%','display': 'inline-block','margin-left':'0.8%'}
    ),
            ]
        ),
      dbc.Row(
        [html.Br()]),     
    dbc.Row([
        dbc.Col([html.H3(id = "tc", style = {'display': 'inline-block'}),
                 html.Br(),
               html.P(id = "title1", style = {'display': 'inline-block'}),dcc.Graph(id='fig',figure = plot_cases('dl',True))] ),
        dbc.Col([
            html.H3(id = "td", style = {'display': 'inline-block'}),
            html.Br(),
            html.P(id = "title2", style = {'display': 'inline-block'}),
            dcc.Graph(id='fig2',figure = plot_deaths('dl',True))
            ])
        ,]),                                                                               
    dbc.Row(
        [html.Br()]),

],style={"height": "100vh"}

)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])
server = app.server
app.layout = html.Div([body])


app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

@app.callback(
    Output('fig', 'figure'),
    Input('st', 'value'),
    Input('cum-act','value'))
def update_figure(st,ca):
    fig1 = plot_cases(st,ca)
    fig1.update_layout(transition_duration=500)
    return fig1

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
    fig3 = plot_total_cases(ca)
    fig3.update_layout(transition_duration=500)
    return fig3

@app.callback(
    Output('fig4', 'figure'),
    Input('cum-act','value'))
def update_figure4(ca):
    fig4 = plot_total_deaths(ca)
    fig4.update_layout(transition_duration=500)
    return fig4
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

@app.callback(
    Output('title1','children'),
    Input('st','value')
    )
def update_output_div5(st):
    return u'Cases in {}'.format(state_dic[st]) 

@app.callback(
    Output('title2','children'),
    Input('st','value')
    )
def update_output_div6(st):
    return u'Deaths in {}'.format(state_dic[st]) 

if __name__ == '__main__':
    app.run_server(debug=True)