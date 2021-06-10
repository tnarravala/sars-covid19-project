#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 09:34:13 2021

@author: thejeswarreddynarravala
"""

import os
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_daq as daq
import pandas as pd
import plotly.graph_objects as go
from datetime import date
from app import app
import time
from Fitting_india_V2 import simulate_combined,simulate_release

import numpy as np

#import matplotlib

#import matplotlib.pyplot as plt
import sys

import datetime


np.set_printoptions(threshold=sys.maxsize)
Geo = 0.98
num_para = 14

# num_threads = 200
num_threads = 1
num_CI = 1000
# num_CI = 5
start_dev = 0

num_threads_dist = 0

# weight of G in initial fitting
theta = 0.7
# weight of G in release fitting
theta2 = 0.8

I_0 = 5
beta_range = (0.1, 100)
gammaE_range = (0.2, 0.3)
alpha_range = (0.1, 0.9)
gamma_range = (0.04, 0.2)
gamma2_range = (0.04, 0.2)
gamma3_range = (0.04, 0.2)
# sigma_range = (0.001, 1)
a1_range = (0.01, 0.5)
a2_range = (0.001, 0.2)
a3_range = (0.01, 0.2)
eta_range = (0.001, 0.95)
c1_fixed = (0.9, 0.9)
c1_range = (0.8, 1)
h_range = (1 / 30, 1 / 14)
Hiding_init_range = (0.1, 0.9)
k_range = (0.1, 2)
k2_range = (0.1, 2)
I_initial_range = (0, 1)
start_date = '2021-02-01'
reopen_date = '2021-03-15'
end_date = '2021-06-09'
release_date = "2021-06-15" #input june 1st,2021 and Aug 1st,2021
release_frac = 1/4 #input
k_drop = 14
p_m = 1
# Hiding = 0.33
delay = 7
change_eta2 = False
size_ext = 180
release_days = 30 #input
fig_row = 5
fig_col = 3
date_range = ["2021-02-10", "2021-09-15"]
states = ['kl', 'dl', 'tg', 'rj', 'hr', 'jk', 'ka', 'la', 'mh', 'pb', 'tn', 'up', 'ap', 'ut', 'or', 'wb', 'py', 'ch',
          'ct', 'gj', 'hp', 'mp', 'br', 'mn', 'mz', 'ga', 'an', 'as', 'jh', 'ar', 'tr', 'nl', 'ml', 'sk', 'dn_dd', 'ld']


paraFile = pd.read_csv(f'fittingV2_{end_date}/paras.csv')
ConfirmFile = pd.read_csv('indian_cases_confirmed_cases.csv')
DeathFile = pd.read_csv('indian_cases_confirmed_deaths.csv')
PopFile = pd.read_csv( 'state_population.csv')

state_dict = {'ap':'Andhra Pradesh',
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

def extend_state2(state, para_row,release_frac, peak_ratio,
                 daily_speed,cum,release_d):
   para_row = list(para_row)[1:]
   [beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta, h, Hiding_init, c1, I_initial, metric1, metric2, r1, r2, reopen_date ] =  para_row

   release_size = min(1 - eta, eta * release_frac)
   

   #print(
    #  f'eta={round(eta, 3)} hiding={round(eta * Hiding_init, 3)} release={round(release_size, 3)} in {state_dict[state]}')

   df = PopFile
   n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']
   df = ConfirmFile
   confirmed = df[df.iloc[:, 0] == state]
   df2 = DeathFile
   death = df2[df2.iloc[:, 0] == state]
   dates = list(confirmed.columns)
   dates = dates[dates.index(start_date):dates.index(end_date) + 1]
   days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in dates]
   confirmed = confirmed.iloc[0].loc[start_date: end_date]
   death = death.iloc[0].loc[start_date: end_date]
   reopen_day = days.index(datetime.datetime.strptime(reopen_date, '%Y-%m-%d'))

   d_confirmed = [confirmed[i] - confirmed[i - 1] for i in range(1, len(confirmed))]
   d_confirmed.insert(0, 0)
   d_death = [death[i] - death[i - 1] for i in range(1, len(death))]
   d_death.insert(0, 0)

   S = [n_0 * eta * (1 - Hiding_init)]
   E = [0]
   I = [n_0 * eta * I_initial * (1 - alpha)]
   A = [n_0 * eta * I_initial * alpha]
   IH = [0]
   IN = [I[-1] * gamma2]
   D = [death[0]]
   R = [0]
   G = [confirmed[0]]
   H = [n_0 * eta * Hiding_init]
   # H = [0]
   size = len(days)
   days_ext = [days[0] + datetime.timedelta(days=i) for i in range(size + size_ext)]
   dates_ext = [d.strftime('%Y-%m-%d') for d in days_ext]

   result, [S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0, betas0] \
      = simulate_combined(size + size_ext, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3,
                          a1, a2, a3, h, Hiding_init, eta, c1, n_0, reopen_day)

   dG0 = [G0[i] - G0[i - 1] for i in range(1, len(G0))]
   dG0.insert(0, 0)
   dD0 = [D0[i] - D0[i - 1] for i in range(1, len(D0))]
   dD0.insert(0, 0)
   peak_dG = 0
   peak_day = 0
 
   # release_day = max(release_day, dates_ext.index('2021-06-01'))
   release_day = dates_ext.index(release_d)

   S = [n_0 * eta * (1 - Hiding_init)]
   E = [0]
   I = [n_0 * eta * I_initial * (1 - alpha)]
   A = [n_0 * eta * I_initial * alpha]
   IH = [0]
   IN = [I[-1] * gamma2]
   D = [death[0]]
   R = [0]
   G = [confirmed[0]]
   H = [n_0 * eta * Hiding_init]

   result, [S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1, HH1, betas1] \
      = simulate_release(size + size_ext, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3,
                         a1, a2, a3, h, Hiding_init, eta, c1, n_0, reopen_day, release_day, release_size, daily_speed)

   dG1 = [G1[i] - G1[i - 1] for i in range(1, len(G1))]
   dG1.insert(0, 0)
   dD1 = [D1[i] - D1[i - 1] for i in range(1, len(D1))]
   dD1.insert(0, 0)

   
   return [state, confirmed, death, G0, D0, G1, D1, release_day]

def extend_state(state, para_row,release_frac, peak_ratio,
                 daily_speed,cum,release_d):

   para_row = list(para_row)[1:]
   [beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta, h, Hiding_init, c1, I_initial, metric1, metric2, r1, r2, reopen_date ] =  para_row

   release_size = min(1 - eta, eta * release_frac)
   

   #print(
    #  f'eta={round(eta, 3)} hiding={round(eta * Hiding_init, 3)} release={round(release_size, 3)} in {state_dict[state]}')

   df = PopFile
   n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']
   df = ConfirmFile
   confirmed = df[df.iloc[:, 0] == state]
   df2 = DeathFile
   death = df2[df2.iloc[:, 0] == state]
   dates = list(confirmed.columns)
   dates = dates[dates.index(start_date):dates.index(end_date) + 1]
   days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in dates]
   confirmed = confirmed.iloc[0].loc[start_date: end_date]
   death = death.iloc[0].loc[start_date: end_date]
   reopen_day = days.index(datetime.datetime.strptime(reopen_date, '%Y-%m-%d'))

   d_confirmed = [confirmed[i] - confirmed[i - 1] for i in range(1, len(confirmed))]
   d_confirmed.insert(0, 0)
   d_death = [death[i] - death[i - 1] for i in range(1, len(death))]
   d_death.insert(0, 0)

   S = [n_0 * eta * (1 - Hiding_init)]
   E = [0]
   I = [n_0 * eta * I_initial * (1 - alpha)]
   A = [n_0 * eta * I_initial * alpha]
   IH = [0]
   IN = [I[-1] * gamma2]
   D = [death[0]]
   R = [0]
   G = [confirmed[0]]
   H = [n_0 * eta * Hiding_init]
   # H = [0]
   size = len(days)
   days_ext = [days[0] + datetime.timedelta(days=i) for i in range(size + size_ext)]
   dates_ext = [d.strftime('%Y-%m-%d') for d in days_ext]

   result, [S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0, betas0] \
      = simulate_combined(size + size_ext, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3,
                          a1, a2, a3, h, Hiding_init, eta, c1, n_0, reopen_day)

   dG0 = [G0[i] - G0[i - 1] for i in range(1, len(G0))]
   dG0.insert(0, 0)
   dD0 = [D0[i] - D0[i - 1] for i in range(1, len(D0))]
   dD0.insert(0, 0)
   peak_dG = 0
   peak_day = 0
 
   # release_day = max(release_day, dates_ext.index('2021-06-01'))
   release_day = dates_ext.index(release_d)

   S = [n_0 * eta * (1 - Hiding_init)]
   E = [0]
   I = [n_0 * eta * I_initial * (1 - alpha)]
   A = [n_0 * eta * I_initial * alpha]
   IH = [0]
   IN = [I[-1] * gamma2]
   D = [death[0]]
   R = [0]
   G = [confirmed[0]]
   H = [n_0 * eta * Hiding_init]

   result, [S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1, HH1, betas1] \
      = simulate_release(size + size_ext, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3,
                         a1, a2, a3, h, Hiding_init, eta, c1, n_0, reopen_day, release_day, release_size, daily_speed)

   dG1 = [G1[i] - G1[i - 1] for i in range(1, len(G1))]
   dG1.insert(0, 0)
   dD1 = [D1[i] - D1[i - 1] for i in range(1, len(D1))]
   dD1.insert(0, 0)

   
   fig = go.Figure()
   fig2 = go.Figure()
   #fig.add_vline(x =rdate,line_dash ='dash')
   if cum == False:
       fig.add_trace(go.Bar(x=days_ext[1:len(d_confirmed)],y = [i for i in d_confirmed[1:]],name="Reported"))
       fig.add_trace(go.Scatter(x=days_ext[1:],y = [i for i in dG0[1:]],name='Original\nProjection'),)
       fig.add_trace(go.Scatter(x=days_ext[1:],y = [i  for i in dG1[1:]],name=f'{round(release_frac * 100)}% release',fill='tonexty'))
       fig2.add_trace(go.Bar(x=days_ext[1:len(d_death)],y = [i  for i in d_death[1:]],name="Reported"))
       fig2.add_trace(go.Scatter(x=days_ext[1:],y = [i for i in dD0[1:]],name='Original\nProjection'))
       fig2.add_trace(go.Scatter(x=days_ext[1:],y = [i  for i in dD1[1:]],name=f'{round(release_frac * 100)}% release',fill='tonexty'))
   else:
       fig.add_trace(go.Bar(x=days,y = [i  for i in confirmed],name="Reported"))
       fig.add_trace(go.Scatter(x=days_ext,y = [i  for i in G1],name=f'{round(release_frac * 100)}% release'))
       fig.add_trace(go.Scatter(x=days_ext,y = [i  for i in G0],name='Original\nProjection'))
       fig2.add_trace(go.Bar(x=days,y = [i  for i in death],name="Reported"))
       fig2.add_trace(go.Scatter(x=days_ext,y = [i  for i in D1],name=f'{round(release_frac * 100)}% release'))
       fig2.add_trace(go.Scatter(x=days_ext,y = [i  for i in D0],name='Original\nProjection'))
   
   
   fig.update_layout(
    autosize=True,
    #title = titlename,
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
   fig.update_layout(showlegend=False)
   fig.update_yaxes(title=None)
   fig.update_xaxes(title=None)
   
   fig2.update_layout(
    autosize=True,
    #title = titlename,
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
   fig2.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
    ))
   fig2.update_layout(showlegend=False)
   fig2.update_yaxes(title=None)
   fig2.update_xaxes(title=None)
   return [fig,fig2]


def extend_india(confirmed, death, G0, D0, G1, D1, release_day,cum,release_frac):
    d_confirmed = [confirmed[i] - confirmed[i - 1] for i in range(1, len(confirmed))]
    d_confirmed.insert(0, 0)
    d_death = [death[i] - death[i - 1] for i in range(1, len(death))]
    d_death.insert(0, 0)

    dG = [G0[i] - G0[i - 1] for i in range(1, len(G0))]
    dG.insert(0, 0)
    dD = [D0[i] - D0[i - 1] for i in range(1, len(D0))]
    dD.insert(0, 0)
    dG2 = [G1[i] - G1[i - 1] for i in range(1, len(G1))]
    dG2.insert(0, 0)
    dD2 = [D1[i] - D1[i - 1] for i in range(1, len(D1))]
    dD2.insert(0, 0)
    
    fig = go.Figure()
    fig2 = go.Figure()
    

    days_ext = [datetime.datetime.strptime(start_date, '%Y-%m-%d') + datetime.timedelta(days=i) for i in range(len(G0))]

    '''ax.axvline(days_ext[release_day], linestyle='dashed', color='tab:red')
    ax2.axvline(days_ext[release_day], linestyle='dashed', color='tab:red')
    ax7.axvline(days_ext[release_day], linestyle='dashed', color='tab:red')
    ax8.axvline(days_ext[release_day], linestyle='dashed', color='tab:red')'''
    if cum == False:
        fig.add_trace(go.Bar(x=days_ext[1:len(d_confirmed)],y=[i for i in d_confirmed[1:]],name="Reported"))
        fig.add_trace(go.Scatter(x=days_ext[1:],y =[i  for i in dG[1:]],name='Original\nProjection'))
        fig.add_trace(go.Scatter(x =days_ext[1:],y=[i  for i in dG2[1:]],name =f'{round(release_frac * 100)}% release',fill='tonexty'))

        fig2.add_trace(go.Bar(x=days_ext[1:len(d_death)],y=[i  for i in d_death[1:]],name="Reported"))
        fig2.add_trace(go.Scatter(x=days_ext[1:],y =[i  for i in dD[1:]],name='Original\nProjection'))
        fig2.add_trace(go.Scatter(x =days_ext[1:],y=[i for i in dD2[1:]],name =f'{round(release_frac * 100)}% release',fill='tonexty' ))
        

    else:
        fig.add_trace(go.Bar(x=days_ext[:len(confirmed)],y=[i  for i in confirmed],name="Reported"))
        fig.add_trace(go.Scatter(x =days_ext,y=[i for i in G1],name =f'{round(release_frac * 100)}% release' ))
        fig.add_trace(go.Scatter(x=days_ext,y =[i  for i in G0],name='Original\nProjection'))
        fig2.add_trace(go.Bar(x=days_ext[:len(confirmed)],y=[i  for i in death],name="Reported"))
        fig2.add_trace(go.Scatter(x =days_ext,y=[i  for i in D1],name =f'{round(release_frac * 100)}% release' ))
        fig2.add_trace(go.Scatter(x=days_ext,y =[i  for i in D0],name='Original\nProjection'))

    
    fig.update_layout(
    autosize=True,
    #title = titlename,
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
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title=None)
    fig.update_xaxes(title=None)
    fig2.update_layout(
    autosize=True,
    #title = titlename,
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
    fig2.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
    ))
    fig2.update_layout(showlegend=False)
    fig2.update_yaxes(title=None)
    fig2.update_xaxes(title=None)
    #print('type of fig',type(fig))
    return fig,fig2

def extedend_state(state,rel_days,rel_frac,rel_date,ind = 0,cum = False):
    para_row = paraFile[paraFile['state']==state].iloc[0]
    fig,fig2 =extend_state(state
                                   ,
                                   para_row, rel_frac, 0.5, 1 / rel_days,cum,rel_date)
    return fig,fig2

def extend_all(rel_days,rel_frac,rel_date,cum = False):
    India_G0 = []
    India_D0 = []
    India_G1 = []
    India_D1 = []
    India_confirmed = []
    India_death = []
    

    for state in states:
            para_row = paraFile[paraFile['state']==state].iloc[0]
            [state, confirmed, death, G0, D0, G1, D1, release_day] =extend_state2(state,
                                    para_row,rel_frac, 0.5, 1 / rel_days,cum,rel_date)
            India_release_day = release_day
            if len(India_G0) == 0:
                    India_G0 = G0.copy()
                    India_D0 = D0.copy()
                    India_G1 = G1.copy()
                    India_D1 = D1.copy()
                    India_confirmed = confirmed.copy()
                    India_death = death.copy()
            else:
                    India_G0 = [India_G0[i] + G0[i] for i in range(len(G0))]
                    India_D0 = [India_D0[i] + D0[i] for i in range(len(G0))]
                    India_G1 = [India_G1[i] + G1[i] for i in range(len(G0))]
                    India_D1 = [India_D1[i] + D1[i] for i in range(len(G0))]
                    India_confirmed = [India_confirmed[i] + confirmed[i] for i in range(len(confirmed))]
                    India_death = [India_death[i] + death[i] for i in range(len(death))]
    
    
    fig,fig2 = extend_india(India_confirmed, India_death, India_G0, India_D0, India_G1, India_D1, India_release_day,cum,rel_frac)
        
    return fig,fig2


ind_fig,ind_fig2= extend_all(release_days,release_frac,release_date)
st_fig,st_fig2 = extedend_state('dl',release_days,release_frac,release_date)


body = dbc.Container([ 

dbc.Row([html.P("Projections of Daily Cases and Deaths",style={'color':'#151516',"font-size":"25px",'font-weight': 'bold'})]),
dbc.Row([html.P("These projections are based on population behaviour in 2021 and can change based on adoption of social distancing measures",style={'color':'#9E12D6',"font-size":"20px"})]),
dbc.Row([html.P("This data on confirmed cases and deaths has been updated on 9th June, 2021",style= {"color":"#151516",'font-size':'20px'}),]),
dbc.Row(
        [
            dbc.Col(html.Label("Release Date ",style = {'font-size':'20px','display': 'inline-block'})),
            dbc.Col(html.Label("Release Fraction ",style = {'font-size':'20px','display': 'inline-block'})),
            dbc.Col(html.Label("Release days ",style = {'font-size':'20px','display': 'inline-block'}))
        ]
        ),
dbc.Row([
    dbc.Col([
dcc.DatePickerSingle(
    id='date-picker-single',
    date=date(2021, 6, 15),
    style  = {'display': 'inline-block','width':'10px', 'height':'10px'}
)]),
    dbc.Col(dcc.Dropdown(
        id='drp_relfrac',
        options=[
            {'label':'25%','value':0.25},
            {'label': '50%', 'value':0.5},
            {'label':'75%','value':0.75},
            {'label':'100%','value':1},
 
        ],
        value=0.25,style = {'color':'black','width':'75%','display': 'inline-block','margin-left':'0.8%'}
    )),
    dbc.Col(dcc.Dropdown(
        id='rel_d',
        options=[
            {'label':'1 week','value':1*7},
            {'label': '2 week', 'value':2*7},
            {'label':'3 week','value':3*7},
            {'label':'4 week','value':4*7},
 
        ],
        value=1*7,style = {'color':'black','width':'75%','display': 'inline-block','margin-left':'0.8%'}
    ))
    ]),
  dbc.Row([
        dbc.Col([html.Br(),
               html.P(["Cases in India"],style = {'color':'green','display': 'inline-block'}),
            dcc.Loading(
            id="loading-2",
            type="default",
            children=html.Div(dcc.Graph(id='fig_india_cases',figure = ind_fig) ))]),
               
        
        dbc.Col([
html.Br(),
            html.P(["Deaths in India"],style = {'color':'red','display': 'inline-block'}),
            
            dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div(dcc.Graph(id='fig_india_deaths',figure = ind_fig2)))
            
            ])
        ,]),

      dbc.Row(
        [html.Br()]),
    
    dbc.Row(
        [
    dcc.Dropdown(
        id='drp_dn',
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
    dbc.Col([
dcc.DatePickerSingle(
    id='date-picker-single-state',
    date=date(2021, 6, 15),
    style  = {'display': 'inline-block','width':'10px', 'height':'10px'}
)]),
    dbc.Col(dcc.Dropdown(
        id='drp_relfrac_state',
        options=[
            {'label':'25%','value':0.25},
            {'label': '50%', 'value':0.5},
            {'label':'75%','value':0.75},
            {'label':'100%','value':1},
 
        ],
        value=0.25,style = {'color':'black','width':'75%','display': 'inline-block','margin-left':'0.8%'}
    )),
    dbc.Col(dcc.Dropdown(
        id='rel_d_state',
        options=[
            {'label':'1 week','value':1*7},
            {'label': '2 week', 'value':2*7},
            {'label':'3 week','value':3*7},
            {'label':'4 week','value':4*7},
 
        ],
        value=1*7,style = {'color':'black','width':'75%','display': 'inline-block','margin-left':'0.8%'}
    ))
    ]),
   dbc.Row(
        [html.Br()]),
    dbc.Row([
        dbc.Col([
               html.P(id = "state_cases", style = {'color':'green','display': 'inline-block'}),
               dcc.Graph(id='fig_state_cases',figure =st_fig)] ),
        dbc.Col([
            html.P(id = "state_deaths", style = {'color':'red','display': 'inline-block'}),
            dcc.Graph(id='fig_state_deaths',figure = st_fig2)
            
            ])
        ,])

,                                                                             
    

],style={"height": "100vh"}

)
@app.callback(
    [Output('fig_state_cases', 'figure'),
    Output('fig_state_deaths', 'figure')],
    Input('drp_dn','value'),
    Input('date-picker-single-state','date'),
    Input('drp_relfrac_state','value'),
    Input('rel_d_state','value'))
def update_figure_l(ca,rel_date,rel_fra,rel_d):
    [fig,fig2] = extedend_state(ca,rel_d,rel_fra,rel_date)
    fig2.update_layout(transition_duration=500)
    return [fig,fig2]


@app.callback(
    Output('state_cases','children'),
    Input('drp_dn','value')
    )
def update_figure_l3(st):
    return u'Cases in {}'.format(state_dict[st]) 

@app.callback(
    Output('state_deaths','children'),
    Input('drp_dn','value')
    )
def update_figure_l4(st):
    return u'Deaths in {}'.format(state_dict[st]) 

@app.callback(
    [Output('fig_india_cases', 'figure'),
    Output('fig_india_deaths', 'figure')],
    Input('date-picker-single','date'),
    Input('drp_relfrac','value'),
    Input('rel_d','value'))
def update_figure_l5(rel_date,rel_fra,rel_d):
    [fig1,fig2] = extend_all(rel_d,rel_fra,rel_date)
    
    return [fig1,fig2]

@app.callback(Output("loading-output-1", "children"), Input("fig_india_deaths", "figure"))
def input_triggers_spinner(figure):
    time.sleep(10)
    return figure

@app.callback(Output("loading-output-2", "children"), Input("fig_india_cases", "figure"))
def input_triggers_spinner2(figure):
    time.sleep(10)
    return figure




#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])
server = app.server
layout = html.Div([body])
