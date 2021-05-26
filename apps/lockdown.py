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
end_date = '2021-05-22'
release_date = "2021-06-01" #input june 1st,2021 and Aug 1st,2021
release_frac = 1/4 #input
k_drop = 14
p_m = 1
# Hiding = 0.33
delay = 7
change_eta2 = False
size_ext = 75
release_days = 30 #input
fig_row = 5
fig_col = 3
date_range = ["2021-02-10", "2021-08-5"]
states = ['kl', 'dl', 'tg', 'rj', 'hr', 'jk', 'ka', 'la', 'mh', 'pb', 'tn', 'up', 'ap', 'ut', 'or', 'wb', 'py', 'ch',
          'ct', 'gj', 'hp', 'mp', 'br', 'mn', 'mz', 'ga', 'an', 'as', 'jh', 'ar', 'tr', 'nl', 'ml', 'sk', 'dn_dd', 'ld']


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

def extend_state(state, ConfirmFile, DeathFile, PopFile, ParaFile, release_frac, peak_ratio,
                 daily_speed,cd,cum,release_d):
   state_path = f'extended/{state}'
   if not os.path.exists(state_path):
      os.makedirs(state_path)
   df = pd.read_csv(ParaFile)
   beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta, h, Hiding_init, c1, I_initial, metric1, metric2, r1, r2, reopen_date  = \
      df.iloc[0]

   release_size = min(1 - eta, eta * release_frac)
   

   #print(
    #  f'eta={round(eta, 3)} hiding={round(eta * Hiding_init, 3)} release={round(release_size, 3)} in {state_dict[state]}')

   df = pd.read_csv(PopFile)
   n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']
   df = pd.read_csv(ConfirmFile)
   confirmed = df[df.iloc[:, 0] == state]
   df2 = pd.read_csv(DeathFile)
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
   release_day = dates_ext.index(release_date)

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
   if cd == "cases" and cum == False:
       fig.add_trace(go.Bar(x=days_ext[1:len(d_confirmed)],y = [i / 1000 for i in d_confirmed[1:]],name="Reported"))
       fig.add_trace(go.Scatter(x=days_ext[1:],y = [i / 1000 for i in dG1[1:]],name=f'{round(release_frac * 100)}% release'))
       fig.add_trace(go.Scatter(x=days_ext[1:],y = [i / 1000 for i in dG0[1:]],name='Original\nProjection'))
       fig.add_vline(days_ext[reopen_day],line_dash ='dash')
   elif cd == "deaths" and cum == False:
       fig.add_trace(go.Bar(x=days_ext[1:len(d_death)],y = [i / 1000 for i in d_death[1:]],name="Reported"))
       fig.add_trace(go.Scatter(x=days_ext[1:],y = [i / 1000 for i in dD1[1:]],name=f'{round(release_frac * 100)}% release'))
       fig.add_trace(go.Scatter(x=days_ext[1:],y = [i / 1000 for i in dD0[1:]],name='Original\nProjection'))
       fig.add_vline(days_ext[reopen_day],line_dash ='dash')
   elif cd == "cases" and cum == True:
       fig.add_trace(go.Bar(x=days,y = [i / 1000 for i in confirmed],name="Reported"))
       fig.add_trace(go.Scatter(x=days_ext,y = [i / 1000 for i in G1],name=f'{round(release_frac * 100)}% release'))
       fig.add_trace(go.Scatter(x=days_ext,y = [i / 1000 for i in G0],name='Original\nProjection'))
       fig.add_vline(days_ext[reopen_day],line_dash ='dash')
   elif cd == "deaths" and cum == True:
       fig.add_trace(go.Bar(x=days,y = [i / 1000 for i in death],name="Reported"))
       fig.add_trace(go.Scatter(x=days_ext,y = [i / 1000 for i in D1],name=f'{round(release_frac * 100)}% release'))
       fig.add_trace(go.Scatter(x=days_ext,y = [i / 1000 for i in D0],name='Original\nProjection'))
       fig.add_vline(days_ext[reopen_day],line_dash ='dash')

   '''data = [S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0, betas0, S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1, HH1, betas1]
   c0 = ['S', 'E', 'I', 'A', 'IH', 'IN', 'D', 'R', 'G', 'H', 'betas', 'S1', 'E1', 'I1', 'A1', 'IH1', 'IN1', 'D1', 'R1',
         'G1', 'H1', 'HH1', 'betas1']
   df = pd.DataFrame(data, columns=dates_ext)
   df.insert(0, 'series', c0)
   df.to_csv(f'{state_path}/sim.csv', index=False)'''
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
   return fig,state, confirmed, death, G0, D0, G1, D1, release_day


def extend_india(confirmed, death, G0, D0, G1, D1, release_day,cd,cum):
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
    

    days_ext = [datetime.datetime.strptime(start_date, '%Y-%m-%d') + datetime.timedelta(days=i) for i in range(len(G0))]

    '''ax.axvline(days_ext[release_day], linestyle='dashed', color='tab:red')
    ax2.axvline(days_ext[release_day], linestyle='dashed', color='tab:red')
    ax7.axvline(days_ext[release_day], linestyle='dashed', color='tab:red')
    ax8.axvline(days_ext[release_day], linestyle='dashed', color='tab:red')'''
    if cd == "cases" and cum == False:
        fig.add_trace(go.Bar(x=days_ext[1:len(d_confirmed)],y=[i / 1000 for i in d_confirmed[1:]],name="Reported"))
        fig.add_trace(go.Scatter(x =days_ext[1:],y=[i / 1000 for i in dG2[1:]],name =f'{round(release_frac * 100)}% release' ))
        fig.add_trace(go.Scatter(x=days_ext[1:],y =[i / 1000 for i in dG[1:]],name='Original\nProjection'))
        fig.add_vline(days_ext[release_day],line_dash ='dash')
        titlename = "Cases in India"
    elif cd == "deaths" and cum == False:
        fig.add_trace(go.Bar(x=days_ext[1:len(d_death)],y=[i / 1000 for i in d_death[1:]],name="Reported"))
        fig.add_trace(go.Scatter(x =days_ext[1:],y=[i / 1000 for i in dD2[1:]],name =f'{round(release_frac * 100)}% release' ))
        fig.add_trace(go.Scatter(x=days_ext[1:],y =[i / 1000 for i in dD[1:]],name='Original\nProjection'))
        fig.add_vline(days_ext[release_day],line_dash ='dash')
        titlename = "Deaths in India"
    elif cd == "cases" and cum == True:
        fig.add_trace(go.Bar(x=days_ext[:len(confirmed)],y=[i / 1000 for i in confirmed],name="Reported"))
        fig.add_trace(go.Scatter(x =days_ext,y=[i / 1000 for i in G1],name =f'{round(release_frac * 100)}% release' ))
        fig.add_trace(go.Scatter(x=days_ext,y =[i / 1000 for i in G0],name='Original\nProjection'))
        fig.add_vline(days_ext[release_day],line_dash ='dash')
        titlename = "Cases in India"
    elif cd == "deaths" and cum == True:
        fig.add_trace(go.Bar(x=days_ext[:len(confirmed)],y=[i / 1000 for i in death],name="Reported"))
        fig.add_trace(go.Scatter(x =days_ext,y=[i / 1000 for i in D1],name =f'{round(release_frac * 100)}% release' ))
        fig.add_trace(go.Scatter(x=days_ext,y =[i / 1000 for i in D0],name='Original\nProjection'))
        fig.add_vline(days_ext[release_day],line_dash ='dash')
        titlename = "Deaths in India"
    
    fig.update_layout(
    autosize=True,
    title = titlename,
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
    return fig

def extedend_state(state,cd,rel_days,rel_frac,rel_date,ind = 0,cum = False):
    fig,state, confirmed, death, G0, D0, G1, D1, release_day =extend_state(state, 'indian_cases_confirmed_cases.csv',
                                   'indian_cases_confirmed_deaths.csv', 'state_population.csv',
                                   f'fittingV2_{end_date}/{state}/para.csv', rel_frac, 0.5, 1 / rel_days,cd,cum,rel_date)
    return fig

def extend_all(cd,rel_days,rel_frac,rel_date,cum = False):
    India_G0 = []
    India_D0 = []
    India_G1 = []
    India_D1 = []
    India_confirmed = []
    India_death = []
    
    for state in states:
            fig,state, confirmed, death, G0, D0, G1, D1, release_day =extend_state(state, 'indian_cases_confirmed_cases.csv',
                                   'indian_cases_confirmed_deaths.csv', 'state_population.csv',
                                   f'fittingV2_{end_date}/{state}/para.csv', rel_frac, 0.5, 1 / rel_days,cd,cum,rel_date)
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
    
    
    fig = extend_india(India_confirmed, India_death, India_G0, India_D0, India_G1, India_D1, India_release_day,cd,cum)
        
    
     

    
    return fig

body = dbc.Container([ 


dbc.Row([html.P("Projections on removal of lockdown coming soon...",style={'color':'#9E12D6',"font-size":"20px"})]),
dbc.Row([
    dbc.Col([html.Label("Release Date ",style = {'font-size':'20px','display': 'inline-block'}),
dcc.DatePickerSingle(
    id='date-picker-single',
    date=date(2021, 6, 1),
    style  = {'display': 'inline-block','width':'10px', 'height':'10px'}
)]),
    dbc.Col([html.Label("Release Fraction",style = {'font-size':'20px','display': 'inline-block'}),
             dcc.Input(id= 'rel_fra',type ='number', min =0.1, max =1,step =0.01,value =0.25,style  = {'display': 'inline-block','width':'50px', 'height':'30px'} )])
    ,
    dbc.Col([html.Label("Release Days",style = {'font-size':'20px','display': 'inline-block'}),
             dcc.Input(id='rel_d',type='number',value=30,min =1, max=90,style  = {'display': 'inline-block','width':'50px', 'height':'30px'})])
    ]),

      dbc.Row(
        [html.Br()]),
    dbc.Row([
   dbc.Col([html.H3(id = "sim_ic", style = {'display': 'inline-block'}),
                 html.Br(),
                 html.P("Cummulative",style = {'display': 'inline-block'}),
                 daq.BooleanSwitch(
                id='bool_cum_cases',
                on=False,
                style = {'display': 'inline-block','size':'20%'}
                        ),
                               html.Br(),
            
              # html.P(id = "sim_ind_title", style = {'color':'green','display': 'inline-block'}),
               dcc.Graph(id='fig_ind_cases',figure = extend_all('cases',30,0.25,release_date))] ),
        dbc.Col([
           # html.H3(id = "sim_i_d", style = {'display': 'inline-block'}),
            html.Br(),
                 html.P("Cummulative",style = {'display': 'inline-block'}),
                 daq.BooleanSwitch(
                id='bool_cum_deaths',
                on=False,
                style = {'display': 'inline-block','size':'20%'}
                        ),
                               html.Br(),
              
            #html.P(id = "sim_ind_title2", style = {'color':'red','display': 'inline-block'}),
            dcc.Graph(id='fig_ind_deaths',figure = extend_all('deaths',30,0.25,release_date))
            
            ]),
    
   ]
        ),
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
        dbc.Col([#html.H3(id = "sim_tc", style = {'display': 'inline-block'}),
                 html.Br(),
                 html.P("Cummulative",style = {'display': 'inline-block'}),
                 daq.BooleanSwitch(
                id='cum_state_cases',
                on=False,
                style = {'display': 'inline-block','size':'20%'}
                        ),
                               html.Br(),
               html.P(id = "state_cases", style = {'color':'green','display': 'inline-block'}),
               dcc.Graph(id='fig_state_cases',figure = extedend_state('dl','cases',release_days,0.25,release_date))] ),
        dbc.Col([
            #html.H3(id = "sim_td", style = {'display': 'inline-block'}),
            html.Br(),
                 html.P("Cummulative",style = {'display': 'inline-block'}),
                 daq.BooleanSwitch(
                id='cum_state_deaths',
                on=False,
                style = {'display': 'inline-block','size':'20%'}
                        ),
                               html.Br(),
            html.P(id = "state_deaths", style = {'color':'red','display': 'inline-block'}),
            dcc.Graph(id='fig_state_deaths',figure = extedend_state('dl','deaths',release_days,0.25,release_date))
            
            ])
        ,])

,                                                                             
    

],style={"height": "100vh"}

)
@app.callback(
    Output('fig_state_cases', 'figure'),
    Input('drp_dn','value'),
    Input('cum_state_cases','on'),
    Input('date-picker-single','date'),
    Input('rel_fra','value'),
    Input('rel_d','value'))
def update_figure_l(ca,cum,rel_date,rel_fra,rel_d):
    fig2 = extedend_state(ca,'cases',rel_d,rel_fra,rel_date,0,cum)
    fig2.update_layout(transition_duration=500)
    return fig2

@app.callback(
    Output('fig_state_deaths', 'figure'),
    Input('drp_dn','value'),
    Input('cum_state_deaths','on'),
    Input('date-picker-single','date'),
    Input('rel_fra','value'),
    Input('rel_d','value'))
def update_figure_l1(ca,cum,rel_date,rel_fra,rel_d):
    fig2 = extedend_state(ca,'deaths',rel_d,rel_fra,rel_date,0,cum)
    fig2.update_layout(transition_duration=500)
    return fig2

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
    Output('fig_ind_cases', 'figure'),
    Input('bool_cum_cases','on'),
    Input('date-picker-single','date'),
    Input('rel_fra','value'),
    Input('rel_d','value'))
def update_figure_l5(ca,rel_date,rel_fra,rel_d):
    fig1 = extend_all('cases',rel_d,rel_fra,rel_date,ca)
    fig1.update_layout(transition_duration=500)
    return fig1

@app.callback(
    Output('fig_ind_deaths', 'figure'),
    Input('bool_cum_deaths','on'),
    Input('date-picker-single','date'),
    Input('rel_fra','value'),
    Input('rel_d','value'))
def update_figure_l6(ca,rel_date,rel_fra,rel_d):
    fig2 = extend_all('deaths',rel_d,rel_fra,rel_date,ca)
    fig2.update_layout(transition_duration=500)
    return fig2





#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])
server = app.server
layout = html.Div([body])
