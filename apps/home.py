import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

from app import app
cases = pd.read_csv("indian_cases_confirmed_cases.csv")
deaths = pd.read_csv("indian_cases_confirmed_deaths.csv")
imp_st = pd.read_csv('cases_deaths_india.csv')
imp_st = imp_st.sort_values('date')
sim_range = ["2021-02-10 18:36:37.3129", "2021-05-07 05:23:22.6871"]
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
    sim_data = pd.read_csv(f'fitting_2021-05-03/{state}/sim.csv')
    sim_data1 = sim_data
    sim_data = sim_data[sim_data['series'] == 'G'].T
    sim_data = sim_data[1:].reset_index()
    sim_data.columns = ['date','G']
    sim_data['date'] = pd.to_datetime(sim_data['date'])
    dates =  sim_data['date']
    if ca == False:
        st = cases.set_index('state')
        col1 = st['2020-01-30']
        st = st.diff(axis = 1)
        st['2020-01-30'] = col1
        st = st.reset_index()
        st = (st[st['state'] == state].T)
        sim_data = sim_data['G'].diff()
        sim_data[0] = 0
        sim_data = sim_data.to_frame()
        sim_data['date'] = dates
        sim_data.columns = ['G','date']
    else:
        st = (cases[cases['state'] == state].T)
    
    #sim_data1 = sim_data1[sim_data1['series'] == 'A'].T
    #sim_data1 = sim_data1[1:].reset_index()
    #sim_data1.columns = ['date','A']
    #sim_data1['date'] = pd.to_datetime(sim_data1['date'])
    dates =  sim_data['date']
    st = st[1:].reset_index()
    st.columns = ['date','cases']
    st['date'] = pd.to_datetime(st['date'])
    st_name = u'Cases in {}'.format(state_dic[state])
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=st['date'],y = st['cases'],name="Actual G"))
    fig.add_trace(go.Scatter(x=sim_data['date'],y = sim_data['G'],name="G"))
    #fig.add_trace(go.Scatter(x=sim_data1['date'],y = sim_data1['A'],name="A"))
    #fig = go.Figure()
    #fig.add_trace(go.Scatter(x=st['date'],y=st['cases'],mode= 'markers',name='Cases'))
    #fig.add_trace(go.Scatter(x=sim_data['date'],y=sim_data['infections'],mode= 'markers',name='I'))
    #fig = make_subplots(rows = 6, cols =6, start_cell = "top-left")
    #fig.add_trace(go.Scatter(x=st['date'],y=st['cases'],mode= 'markers'))
    #fig = px.scatter(st, x='date', y='cases')
    #fig = go.Figure()
    #fig.add_trace(go.scatter(x=sim_data['date'],y=sim_data['infections'],mode ="lines",name="infections"))
    #fig.add_trace(go.scatter(x=st['date'],y=st['cases'],mode ="lines"))
    #fig.add_trace()
    fig.update_layout(
    autosize=True,
    #title = st_name,
    margin = dict(l=40, r=40, t=10, b=40 ),
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
    sim_data = pd.read_csv(f'fitting_2021-05-03/{state}/sim.csv')
    sim_data = sim_data[sim_data['series'] == 'D'].T
    sim_data = sim_data[1:].reset_index()
    sim_data.columns = ['date','D']
    sim_data['date'] = pd.to_datetime(sim_data['date'])
    dates =  sim_data['date']
    if ca == False:
        st = deaths.set_index('state')
        col1 = st['2020-01-30']
        st = st.diff(axis = 1)
        st['2020-01-30'] = col1
        st = st.reset_index()
        st = (st[st['state'] == state].T)
        sim_data = sim_data['D'].diff()
        sim_data[0] = 0
        sim_data = sim_data.to_frame()
        sim_data['date'] = dates
        sim_data.columns = ['D','date']
    else:
        st = (deaths[deaths['state'] == state].T)
    st = st[1:].reset_index()
    st.columns = ['date','deaths']
    st['date'] = pd.to_datetime(st['date'])
    #fig = go.Figure()
    #fig.add_trace(go.Scatter(x=st['date'],y=st['deaths'],mode= 'markers',name=f'{state_dic[state]}'))
    st_name = u'Deaths in {}'.format(state_dic[state])
    #fig = px.bar(st, x='date', y='deaths')
    fig = go.Figure()
    fig.add_trace(go.Bar(x=st['date'],y = st['deaths'],name="Actual D"))
    fig.add_trace(go.Scatter(x=sim_data['date'],y = sim_data['D'],name="D"))
    fig.update_layout(
    autosize=True,
    #title =  st_name,

    margin = dict(l=40, r=40, t=10, b=40 ),
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
            range=date_range,
            
            
            
        ),
        type="date",
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


def plot_all_states(state):
    ct = imp_st[imp_st['state'] == state]
    sim_data = pd.read_csv(f'fitting_2021-05-03/{state}/sim.csv')
    sim_data = sim_data[sim_data['series'] == 'G'].T
    sim_data = sim_data[1:].reset_index()
    sim_data.columns = ['date','G']
    sim_data['date'] = pd.to_datetime(sim_data['date'])
    dates =  sim_data['date']
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ct['date'],y = ct['confirmed_cases']))
    #fig.add_trace(go.Scatter(x=sim_data['date'],y = sim_data['G']))
    #fig = px.bar(ct, x='date', y='confirmed_cases')
    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                   opacity=0.3)
    fig.update_layout(
    autosize=True,
    title = state_dic[state],
    margin = dict(l=20, r=10, t=27, b=10 ),
    width=300,
    height=200,
    yaxis = dict(
       #range = [0,100] ,
       #rangemode="tozero",
        autorange=True,
        title_text='cases',
       # titlefont=dict(size=10),
    ),
    xaxis=dict(
        #title_text = "date",
        autorange=True,
        range=date_range,
        )
    )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(visible=False, showticklabels=False)
    #fig.update_xaxes(visible=False, showticklabels=False)
    return fig

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']





body = dbc.Container([ 

  dbc.Row([
        dbc.Col([html.H3(id = "tsc", style = {'display': 'inline-block'}),
                 html.Br(),
                 html.P("Cummulative",style = {'display': 'inline-block'}),
                 daq.BooleanSwitch(
                id='cum-tc',
                on=False,
                style = {'display': 'inline-block','size':'20%'}
                        ),
                 dcc.Graph(id="fig3",figure = plot_total_cases('Daily new cases'))]),
        
        
        dbc.Col([html.H3(id = "tsd", style = {'display': 'inline-block'}),
                 html.Br(),
                 html.P("Cummulative",style = {'display': 'inline-block'}),
                 daq.BooleanSwitch(
                id='cum-td',
                on=False,
                style = {'display': 'inline-block','size':'20%'}
                        ),
                  dcc.Graph(id="fig4",figure = plot_total_deaths('Daily new cases'))])
        ,],align='center',justify = "center"),
  
dbc.Row(
        [html.Br()]),

dbc.Row(
        [
    dcc.Dropdown(
        id='st',
        options=[
            {'label':'Andaman and Nicobar','value':'an'},
            {'label': 'Andhra Pradesh', 'value': 'ap'},
            {'label':'Arunachal Pradesh','value':'ar'},
            {'label':'Assam','value':'as'},
            {'label':'Bihar','value':'br'},
            {'label':'Chandigarh','value':'ch'},
            {'label':'Chattisgarh','value':'ct'},
            {'label':'Daman and Diu','value':'dn_dd'},
            {'label':'Delhi','value':'dl'},
            {'label':'Goa','value':'ga'},
            {'label':'Gujarat','value':'gj'},
            {'label':'Haryana','value':'hr'},
            {'label':'Himachal Pradesh','value':'hp'},
            {'label':'Jammu and Kashmir','value':'jk'},
            {'label':'Jharkhand','value':'jh'},
            {'label':'Karnataka','value':'ka'},
            {'label': 'Kerala', 'value': 'kl'},
            {'label':'Ladakh','value':'ld'},
            {'label':'Lakshdweep','value':'la'},
            {'label': 'Madhya Pradesh', 'value': 'mp'},
            {'label':'Maharastra','value':'mh'},
            {'label':'Manipur','value':'mn'},
            {'label':'Meghalaya','value':'ml'},
            {'label':'Mizoram','value':'mz'},
            {'label':'Nagaland','value':'nl'},
            {'label':'Odisha','value':'or'},
            {'label':'Puducherry','value':'py'},
            {'label':'Punjab','value':'pb'},
            {'label':'Rajesthan','value':'rj'},
            {'label':'Sikkim','value':'sk'},
            {'label':'Tamil Nadu','value':'tn'},
            {'label':'Telangana','value':'tg'},
            {'label':'Tripura','value':'tr'},
            {'label':'Uttarakhand','value':'ut'},
            {'label':'Uttar Pradesh','value':'up'},
            {'label':'West Bengal','value':'wb'},
 
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
                 html.P("Cummulative",style = {'display': 'inline-block'}),
                 daq.BooleanSwitch(
                id='cum-c',
                on=False,
                style = {'display': 'inline-block','size':'20%'}
                        ),
                               html.Br(),
               html.P(id = "title1", style = {'color':'green','display': 'inline-block'}),dcc.Graph(id='fig',figure = plot_cases('dl',True))] ),
        dbc.Col([
            html.H3(id = "td", style = {'display': 'inline-block'}),
            html.Br(),
                 html.P("Cummulative",style = {'display': 'inline-block'}),
                 daq.BooleanSwitch(
                id='cum-d',
                on=False,
                style = {'display': 'inline-block','size':'20%'}
                        ),
                               html.Br(),
            html.P(id = "title2", style = {'color':'red','display': 'inline-block'}),
            dcc.Graph(id='fig2',figure = plot_deaths('dl',True))
            
            ])
        ,]),                                                                               
    dbc.Row(
        [html.Br()]),
    dbc.Row(
        [
            html.H2("Cummulative cases for each state in India")
            ]
        ),
    
    dbc.Row(
        [
        
            dbc.Col(dcc.Graph(id='fig5',figure = plot_all_states('an')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig6',figure = plot_all_states('ap')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig7',figure = plot_all_states('ar')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig8',figure = plot_all_states('as')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig9',figure = plot_all_states('br')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig10',figure = plot_all_states('ch')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig11',figure = plot_all_states('ct')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig12',figure = plot_all_states('dn_dd')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig13',figure = plot_all_states('dl')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig14',figure = plot_all_states('ga')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig15',figure = plot_all_states('gj')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig16',figure = plot_all_states('hr')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig17',figure = plot_all_states('hp')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig18',figure = plot_all_states('jk')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig19',figure = plot_all_states('jh')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig20',figure = plot_all_states('ka')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig21',figure = plot_all_states('kl')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig22',figure = plot_all_states('ld')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig23',figure = plot_all_states('la')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig24',figure = plot_all_states('mp')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig25',figure = plot_all_states('mh')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig26',figure = plot_all_states('ml')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig27',figure = plot_all_states('mn')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig28',figure = plot_all_states('mz')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig29',figure = plot_all_states('nl')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig30',figure = plot_all_states('or')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig31',figure = plot_all_states('py')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig32',figure = plot_all_states('pb')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig33',figure = plot_all_states('rj')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig34',figure = plot_all_states('sk')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig35',figure = plot_all_states('tn')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig36',figure = plot_all_states('tg')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig37',figure = plot_all_states('tr')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig38',figure = plot_all_states('ut')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig39',figure = plot_all_states('up')),width=6, lg=3),
            dbc.Col(dcc.Graph(id='fig40',figure = plot_all_states('wb')),width=6, lg=3)
            
            ]
        )

],style={"height": "100vh"}

)

#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])
server = app.server
layout = html.Div([body])


app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

@app.callback(
    Output('fig', 'figure'),
    Input('st', 'value'),
    Input('cum-c','on'))
def update_figure(st,ca):
    fig1 = plot_cases(st,ca)
    fig1.update_layout(transition_duration=500)
    return fig1

@app.callback(
    Output('fig2', 'figure'),
    Input('st', 'value'),
    Input('cum-d','on'))
def update_figure2(st,ca):
    fig2 = plot_deaths(st,ca)
    fig2.update_layout(transition_duration=500)
    return fig2

@app.callback(
    Output('fig3', 'figure'),
    Input('cum-tc','on'))
def update_figure3(ca):
    fig3 = plot_total_cases(ca)
    fig3.update_layout(transition_duration=500)
    return fig3

@app.callback(
    Output('fig4', 'figure'),
    Input('cum-td','on'))
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

#app.config.suppress_callback_exceptions = True

'''if __name__ == '__main__':
    app.run_server(debug=True)'''