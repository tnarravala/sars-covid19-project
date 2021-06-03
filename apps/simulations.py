#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:33:13 2021

@author: thejeswarreddynarravala
"""

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
india_cases= pd.read_csv("india_cases_diff.csv")
india_deaths= pd.read_csv("india_deaths_diff.csv")
state_dic = {'ap':'Andhra Pradesh',
             'dl':'Delhi',
             'mp':'Madhya Pradesh',
             'kl':'Kerala',
             'up':'Uttar Pradesh',
             'mh':'Maharastra',
             'br':'Bihar',
             'wb':'West Bengal',
             'tn':'Tamil Nadu',
             'rj':'Rajasthan',
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

date_range = ["2021-02-10", "2021-09-15"]

def plot_cases(state,ca):
    sim_data = pd.read_csv(f'extended/{state}/sim.csv')
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

    st = st[1:].reset_index()
    st.columns = ['date','cases']
    st['date'] = pd.to_datetime(st['date'])
    st = st[st['date'] > '2021-01-31']
    st['mv3'] = st.iloc[:,1].rolling(window=7).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=st['date'],y = st['cases'],name="Actual"))
    fig.add_trace(go.Scatter(x=sim_data['date'],y = sim_data['G'],name="Projections"))
    fig.add_trace(go.Scatter(x=st['date'],y = st['mv3'],name="7-Mav",line = dict(shape = 'linear', color = '#0000FF', dash = 'dash')))
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
    fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
    ))
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

    ),
    )
    #fig.update_layout(showlegend=False)
    fig.update_yaxes(title=None)
    fig.update_xaxes(title=None)
    return fig

def plot_deaths(state,ca):
    sim_data = pd.read_csv(f'extended/{state}/sim.csv')
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
    st = st[st['date'] > '2021-01-31']
    st['mv3'] = st.iloc[:,1].rolling(window=7).mean()
    #fig = go.Figure()
    #fig.add_trace(go.Scatter(x=st['date'],y=st['deaths'],mode= 'markers',name=f'{state_dic[state]}'))
    #st_name = u'Deaths in {}'.format(state_dic[state])
    #fig = px.bar(st, x='date', y='deaths')
    fig = go.Figure()
    fig.add_trace(go.Bar(x=st['date'],y = st['deaths'],name="Actual"))
    fig.add_trace(go.Scatter(x=sim_data['date'],y = sim_data['D'],name="Projections"))
    fig.add_trace(go.Scatter(x=st['date'],y = st['mv3'],name="7-Mav",line = dict(shape = 'linear', color = '#0000FF', dash = 'dash')))
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
    ),
    )
    fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
    ))
    #fig.update_layout(showlegend=False)
    fig.update_yaxes(title=None)
    fig.update_xaxes(title=None)
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
    ind = ind[ind['date'] > '2021-01-31']
    tc = india_cases[india_cases['date'] > '2021-01-31']
    ind['mv3'] = ind.iloc[:,1].rolling(window=7).mean()
    fig = go.Figure()
    #fig.add_trace(go.Scatter(x=ind['date'],y=ind['sum'],mode= 'markers'))
    #fig = px.bar(ind, x='date', y='sum')
    fig.add_trace(go.Bar(x=ind['date'],y=ind['sum'],name='Actual'))
    
    if ca == True: 
        fig.add_trace(go.Scatter(x=tc['date'],y=tc['cases'],name='Projections'))
    else:
        fig.add_trace(go.Scatter(x=tc['date'],y=tc['diff'],name='Projections'))
    fig.add_trace(go.Scatter(x=ind['date'],y=ind['mv3'],name='7-Mav',line = dict(shape = 'linear', color = '#0000FF', dash = 'dash')))
    fig.update_layout(
    autosize=True,
    title = "Cases in India",
    margin = dict(l=40, r=40, t=40, b=40 ),
    width=500,
    height=400,
    yaxis = dict(
       #range = [0,100] ,
       rangemode="tozero",
        autorange=True,
        title_text='cases',
        titlefont=dict(size=10),
    ),
    xaxis=dict(
        title_text = "date",
        autorange=True,
        range=date_range,
    ),
    )
    fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
    ))
    #fig.update_layout(showlegend=False)
    fig.update_yaxes(title=None)
    fig.update_xaxes(title=None)
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
    ind = ind[ind['date'] > '2021-01-31']
    tc = india_deaths[india_deaths['date'] > '2021-01-31']
    ind['mv3'] = ind.iloc[:,1].rolling(window=7).mean()
    fig = go.Figure()
    #fig.add_trace(go.Scatter(x=ind['date'],y=ind['sum'],mode= 'markers'))
    fig.add_trace(go.Bar(x=ind['date'],y=ind['sum'],name='Actual'))
    if ca == True: 
        fig.add_trace(go.Scatter(x=tc['date'],y=tc['deaths'],name='Projections'))
    else:
        fig.add_trace(go.Scatter(x=tc['date'],y=tc['diff'],name='Projections'))
    fig.add_trace(go.Scatter(x=ind['date'],y=ind['mv3'],name='7-Mav',line = dict(shape = 'linear', color = '#0000FF', dash = 'dash')))
    #fig.add_trace(go.Scatter(x=cum_pro['date'],y=cum_pro['deaths'],name='D'))
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
        title=None
    ),
    )
    fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
    ))
    #fig.update_layout(showlegend=False)
    fig.update_yaxes(title=None)
    fig.update_xaxes(title=None)
    #fig.update_yaxes(visible=True, showticklabels=True, title=False)
    #fig.update_xaxes(visible=False, showticklabels=True)
    return fig

body = dbc.Container([ 

dbc.Row([html.P("Projections for infections and deaths in Indian States and  for overall India are based in part on the model described in the following paper: ",style= {"color":"#151516","font-size":"20px"}),]),
dbc.Row([html.P(dcc.Link("Hidden Parameters Impacting Resurgence of SARS-CoV-2 Pandemic",href="https://www.medrxiv.org/content/10.1101/2021.01.15.20248217v1",target="_blank",style = {"color":"#6211FF","font-size":"20px"}))]),
dbc.Row([   
        html.Label(['Projections on removal of lockdown can be found on this link ---> ', 
        html.A('here', href='https://sars-covid-tracker-india.herokuapp.com/lockdown',style = {"color":"#E60B1F",'font-size':'20px'})],style={"color":"#151516",'font-size':'20px'})
             ]),
dbc.Row([html.P("Computing is provided by Chameleon Cloud, sponsored by NSF-USA",style = {"font-size":"10px"})]),
dbc.Row([html.P("This data on confirmed cases and deaths has been updated on 1st June, 2021",style= {"color":"#151516",'font-size':'20px'}),]),
      dbc.Row(
        [html.Br()]),
    dbc.Row([
   dbc.Col([html.H3(id = "sim_ic", style = {'display': 'inline-block'}),
                 html.Br(),
                 html.P("Cummulative",style = {'display': 'inline-block'}),
                 daq.BooleanSwitch(
                id='sim_cum-ic',
                on=False,
                style = {'display': 'inline-block','size':'20%'}
                        ),
                               html.Br(),
            
              # html.P(id = "sim_ind_title", style = {'color':'green','display': 'inline-block'}),
               dcc.Graph(id='sim_i_fig',figure = plot_total_cases(True))] ),
        dbc.Col([
           # html.H3(id = "sim_i_d", style = {'display': 'inline-block'}),
            html.Br(),
                 html.P("Cummulative",style = {'display': 'inline-block'}),
                 daq.BooleanSwitch(
                id='sim_cum-i_d',
                on=False,
                style = {'display': 'inline-block','size':'20%'}
                        ),
                               html.Br(),
              
            #html.P(id = "sim_ind_title2", style = {'color':'red','display': 'inline-block'}),
            dcc.Graph(id='sim_i_fig2',figure = plot_total_deaths(True))
            
            ]),
    
   ]
        ),
    dbc.Row(
        [
    dcc.Dropdown(
        id='sim_st',
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
    dbc.Row([
        dbc.Col([html.H3(id = "sim_tc", style = {'display': 'inline-block'}),
                 html.Br(),
                 html.P("Cummulative",style = {'display': 'inline-block'}),
                 daq.BooleanSwitch(
                id='sim_cum-c',
                on=False,
                style = {'display': 'inline-block','size':'20%'}
                        ),
                               html.Br(),
               html.P(id = "sim_title1", style = {'color':'green','display': 'inline-block'}),dcc.Graph(id='sim_fig',figure = plot_cases('dl',True))] ),
        dbc.Col([
            html.H3(id = "sim_td", style = {'display': 'inline-block'}),
            html.Br(),
                 html.P("Cummulative",style = {'display': 'inline-block'}),
                 daq.BooleanSwitch(
                id='sim_cum-d',
                on=False,
                style = {'display': 'inline-block','size':'20%'}
                        ),
                               html.Br(),
            html.P(id = "sim_title2", style = {'color':'red','display': 'inline-block'}),
            dcc.Graph(id='sim_fig2',figure = plot_deaths('dl',True))
            
            ])
        ,])

,                                                                             
    

],style={"height": "100vh"}

)

#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])
server = app.server
layout = html.Div([body])

@app.callback(
    Output('sim_fig', 'figure'),
    Input('sim_st', 'value'),
    Input('sim_cum-c','on'))
def update_figure_sim(st,ca):
    fig1 = plot_cases(st,ca)
    fig1.update_layout(transition_duration=500)
    return fig1

@app.callback(
    Output('sim_fig2', 'figure'),
    Input('sim_st', 'value'),
    Input('sim_cum-d','on'))
def update_figure_sim2(st,ca):
    fig2 = plot_deaths(st,ca)
    fig2.update_layout(transition_duration=500)
    return fig2

@app.callback(
    Output('sim_title1','children'),
    Input('sim_st','value')
    )
def update_output_divsim1(st):
    return u'Cases in {}'.format(state_dic[st]) 

@app.callback(
    Output('sim_title2','children'),
    Input('sim_st','value')
    )
def update_output_divsim2(st):
    return u'Deaths in {}'.format(state_dic[st]) 

@app.callback(
    Output('sim_i_fig', 'figure'),
    Input('sim_cum-ic','on'))
def update_figure_sim3(ca):
    fig1 = plot_total_cases(ca)
    fig1.update_layout(transition_duration=500)
    return fig1

@app.callback(
    Output('sim_i_fig2', 'figure'),
    Input('sim_cum-i_d','on'))
def update_figure_sim4(ca):
    fig2 = plot_total_deaths(ca)
    fig2.update_layout(transition_duration=500)
    return fig2

@app.callback(
    Output('sim_ind_title1','children'),
    Input('sim_st','value')
    )
def update_output_divsim3(st):
    return u'Cases in {}'.format(state_dic[st]) 

@app.callback(
    Output('sim_ind_title2','children'),
    Input('sim_st','value')
    )
def update_output_divsim4(st):
    return u'Deaths in {}'.format(state_dic[st]) 