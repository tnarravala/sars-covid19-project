#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 09:34:13 2021

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
from dateutil.relativedelta import relativedelta
from app import app
from datetime import datetime,time

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

date_range = ["2021-02-10", "2021-07-1"]

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
    fig.add_trace(go.Bar(x=st['date'],y = st['cases'],name="Actual G"))
    fig.add_trace(go.Scatter(x=sim_data['date'],y = sim_data['G'],name="G"))
    fig.add_trace(go.Scatter(x=st['date'],y = st['mv3'],name="mv3"))
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

    ),
    )
    fig.update_layout(showlegend=False)
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
    fig.add_trace(go.Bar(x=st['date'],y = st['deaths'],name="Actual D"))
    fig.add_trace(go.Scatter(x=sim_data['date'],y = sim_data['D'],name="D"))
    fig.add_trace(go.Scatter(x=st['date'],y = st['mv3'],name="mv3"))
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
    fig.update_layout(showlegend=False)
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
    fig = go.Figure()
    #fig.add_trace(go.Scatter(x=ind['date'],y=ind['sum'],mode= 'markers'))
    #fig = px.bar(ind, x='date', y='sum')
    fig.add_trace(go.Bar(x=ind['date'],y=ind['sum'],name='Actual G'))
    if ca == True: 
        fig.add_trace(go.Scatter(x=tc['date'],y=tc['cases'],name='G'))
    else:
        fig.add_trace(go.Scatter(x=tc['date'],y=tc['diff'],name='G'))
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
    ),
    )
    fig.update_layout(showlegend=False)
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
    fig = go.Figure()
    #fig.add_trace(go.Scatter(x=ind['date'],y=ind['sum'],mode= 'markers'))
    fig.add_trace(go.Bar(x=ind['date'],y=ind['sum'],name='Actual D'))
    if ca == True: 
        fig.add_trace(go.Scatter(x=tc['date'],y=tc['deaths'],name='D'))
    else:
        fig.add_trace(go.Scatter(x=tc['date'],y=tc['diff'],name='D'))
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
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title=None)
    fig.update_xaxes(title=None)
    #fig.update_yaxes(visible=True, showticklabels=True, title=False)
    #fig.update_xaxes(visible=False, showticklabels=True)
    return fig

daterange = pd.date_range(start='2017',end='2018',freq='W')
def unix_time_millis(dt):
    ''' Convert datetime to unix timestamp '''
    return int(time.mktime(dt.timetuple()))

def get_marks_from_start_end(start, end, Nth=100):
    ''' Returns the marks for labeling. 
        Every Nth value will be used.
    '''

    result = {}
    for i, date in enumerate(daterange):
        if(i%Nth == 1):
            # Append value to dict
            result[unix_time_millis(date)] = str(date.strftime('%Y-%m-%d'))

    return result

#d1 = unix_time_millis(date_range.datetime.min())
body = dbc.Container([ 


dbc.Row([html.P("Projections on removal of lockdown coming soon...",style={'color':'#9E12D6',"font-size":"20px"})]),\


    dbc.Row(
        [
    dbc.Col(dcc.Dropdown(
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
    )),

    
            ]
        ),
                                                                              
    

],style={"height": "100vh"}

)

#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])
server = app.server
layout = html.Div([body])
