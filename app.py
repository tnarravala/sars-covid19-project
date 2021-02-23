import os

import numpy as np
import pandas as pd
from SIRfunctions import SEIARG, SEIARG_fixed, computeBeta_combined
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

city_vaccine_rate = 0.0015
city_test_rate = 0.02
school_beta_multiplier = 1
city_beta_multiplier = 1
log_scale = False

fix_school_beta = True
daytime_fraction = 1 / 2

test_cost = 25
hosp_cost = [5500, 8500, 11600]


class Node:

	def pop(self, day):
		return self.S[day] + self.E[day] + self.I[day] + self.A[day] + self.IH[day] + self.IN[day] + self.D[day] + \
		       self.R[day] + self.Q[day]

	def read_para(self, SimFolder):
		ParaFile = f'{SimFolder}/{self.state}/para.csv'
		df = pd.read_csv(ParaFile)
		beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, metric3, r1, r2, r3 = \
			df.iloc[0]
		return [beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, h, Hiding_init, k, k2, eta, c1]

	def read_pop(self, PopFile):
		df = pd.read_csv(PopFile)
		n_0 = df[df.iloc[:, 0] == self.state].iloc[0]['POP']
		return n_0

	def __init__(self, state, SimFolder, PopFile, initial_infection, test_rate, vaccine_rate):
		self.state = state
		self.t_rate = test_rate
		self.v_rate = vaccine_rate
		self.n_0 = self.read_pop(PopFile)

		[self.beta,
		 self.gammaE,
		 self.alpha,
		 self.gamma,
		 self.gamma2,
		 self.gamma3,
		 self.a1,
		 self.a2,
		 self.a3,
		 self.h,
		 self.Hiding_init,
		 self.k,
		 self.k2,
		 self.eta,
		 self.c1] = self.read_para(SimFolder)

		self.S = [self.n_0 * self.eta]
		self.E = [0]
		self.I = [initial_infection]
		self.A = [0]
		self.IH = [0]
		self.IN = [0]
		self.Q = [0]
		self.D = [0]
		self.R = [0]
		self.G = [initial_infection]
		self.H = [self.Hiding_init * self.n_0 * self.eta]
		self.Betas = []
		self.GH = [0]
		self.GT = [0]
		self.GV = [0]
		self.GI = [initial_infection]

	def commute_out(self, commuter):
		self.S[-1] -= commuter.S[-1]
		self.E[-1] -= commuter.E[-1]
		self.I[-1] -= commuter.I[-1]
		self.A[-1] -= commuter.A[-1]
		return commuter

	def vac_with_com(self, day, com):
		self.GV[day] += self.S[day] * self.v_rate
		com.GV[day] += com.S[day] * self.v_rate

		self.S[day] -= self.S[day] * self.v_rate
		com.S[day] -= com.S[day] * self.v_rate

	def vac(self, day):
		self.GV[day] += self.S[day] * self.v_rate
		self.S[day] -= self.S[day] * self.v_rate

	def sim_with_com(self, day, com, fixed_beta, frac):
		if fixed_beta:
			beta = self.beta
		else:
			beta = computeBeta_combined(self.beta, self.eta, self.n_0, self.S[-1] + com.S[-1], self.I[-1] + com.I[-1],
			                            0, self.c1, 0)
		dS = - beta * self.S[-1] * (self.I[-1] + self.A[-1] + com.I[-1] + com.A[-1]) / self.n_0 * frac
		dS_c = - beta * com.S[-1] * (self.I[-1] + self.A[-1] + com.I[-1] + com.A[-1]) / self.n_0 * frac

		dE = (beta * self.S[-1] * (self.I[-1] + self.A[-1] + com.I[-1] + com.A[-1]) / self.n_0 - self.gammaE * self.E[
			-1]) * frac
		dE_c = (beta * com.S[-1] * (self.I[-1] + self.A[-1] + com.I[-1] + com.A[-1]) / self.n_0 - self.gammaE * com.E[
			-1]) * frac

		dI = ((1 - self.alpha) * self.gammaE * self.E[-1] - (self.gamma + self.gamma2) * self.I[-1]) * frac
		dI_c = ((1 - self.alpha) * self.gammaE * com.E[-1] - (self.gamma + self.gamma2) * com.I[-1]) * frac

		dA = (self.alpha * self.gammaE * self.E[-1] - self.gamma3 * self.A[-1]) * frac
		dA_c = (self.alpha * self.gammaE * com.E[-1] - self.gamma3 * com.A[-1]) * frac

		dQ = - (self.gamma + self.gamma2) * self.Q[-1] * frac
		dQ_c = - (self.gamma + self.gamma2) * com.Q[-1] * frac

		dIH = (self.gamma * (self.I[-1] + self.Q[-1]) - (self.a1 + self.a2) * self.IH[-1]) * frac
		dIH_c = (self.gamma * (com.I[-1] + com.Q[-1]) - (self.a1 + self.a2) * com.IH[-1]) * frac

		dIN = (self.gamma2 * (self.I[-1] + self.Q[-1]) - self.a3 * self.IN[-1]) * frac
		dIN_c = (self.gamma2 * (com.I[-1] + com.Q[-1]) - self.a3 * com.IN[-1]) * frac

		dD = self.a2 * self.IH[-1] * frac
		dD_c = self.a2 * com.IH[-1] * frac

		dR = (self.a1 * self.IH[-1] + self.a3 * self.IN[-1] + self.gamma3 * self.A[-1]) * frac
		dR_c = (self.a1 * com.IH[-1] + self.a3 * com.IN[-1] + self.gamma3 * com.A[-1]) * frac

		dG = (1 - self.alpha) * self.gammaE * self.E[-1] * frac
		dG_c = (1 - self.alpha) * self.gammaE * com.E[-1] * frac

		dGH = self.gamma * (self.I[-1] + self.Q[-1]) * frac
		dGH_c = self.gamma * (com.I[-1] + com.Q[-1]) * frac

		dGI = (1 - self.alpha) * self.gammaE * self.E[-1] * frac
		dGI_c = (1 - self.alpha) * self.gammaE * com.E[-1] * frac

		self.S.append(self.S[-1] + dS)
		com.S.append(com.S[-1] + dS_c)

		self.E.append(self.E[-1] + dE)
		com.E.append(com.E[-1] + dE_c)

		self.I.append(self.I[-1] + dI)
		com.I.append(com.I[-1] + dI_c)

		self.A.append(self.A[-1] + dA)
		com.A.append(com.A[-1] + dA_c)

		self.Q.append(self.Q[-1] + dQ)
		com.Q.append(com.Q[-1] + dQ_c)

		self.IH.append(self.IH[-1] + dIH)
		com.IH.append(com.IH[-1] + dIH_c)

		self.IN.append(self.IN[-1] + dIN)
		com.IN.append(com.IN[-1] + dIN_c)

		self.D.append(self.D[-1] + dD)
		com.D.append(com.D[-1] + dD_c)

		self.R.append(self.R[-1] + dR)
		com.R.append(com.R[-1] + dR_c)

		self.G.append(self.G[-1] + dG)
		com.G.append(com.G[-1] + dG_c)

		self.GH.append(self.GH[-1] + dGH)
		com.GH.append(com.GH[-1] + dGH_c)

		self.GT.append(self.GT[-1])
		com.GT.append(com.GT[-1])

		self.GV.append(self.GV[-1])
		com.GV.append(com.GV[-1])

		self.GI.append(self.GI[-1] + dGI)
		com.GI.append(com.GI[-1] + dGI_c)

	def sim(self, day, fixed_beta, frac):
		if fixed_beta:
			delta = SEIARG_fixed(day,
			                     [self.S[-1], self.E[- 1], self.I[- 1], self.A[- 1], self.IH[- 1], self.IN[- 1],
			                      self.D[- 1], self.R[- 1], self.G[- 1], self.beta, self.gammaE, self.alpha, self.gamma,
			                      self.gamma2, self.gamma3, self.a1, self.a2, self.a3, self.eta, self.n_0, self.c1,
			                      self.H[-1], self.H[0]])
		else:
			delta = SEIARG(day,
			               [self.S[-1], self.E[- 1], self.I[- 1], self.A[- 1], self.IH[- 1], self.IN[- 1], self.D[- 1],
			                self.R[- 1], self.G[- 1], self.beta, self.gammaE, self.alpha, self.gamma, self.gamma2,
			                self.gamma3, self.a1, self.a2, self.a3, self.eta, self.n_0, self.c1, self.H[-1], self.H[0]])

		self.GI.append(self.GI[-1] + (1 - self.alpha) * self.gammaE * self.E[-1] * frac)
		self.S.append(self.S[-1] + delta[0] * frac)
		self.E.append(self.E[-1] + delta[1] * frac)
		self.I.append(self.I[-1] + delta[2] * frac)
		self.A.append(self.A[-1] + delta[3] * frac)
		self.IH.append(self.IH[-1] + delta[4] * frac)
		self.IN.append(self.IN[-1] + delta[5] * frac)
		self.D.append(self.D[-1] + delta[6] * frac)
		self.R.append(self.R[-1] + delta[7] * frac)
		self.G.append(self.G[-1] + delta[8] * frac)
		self.H.append(self.H[-1])
		self.Betas.append(delta[9])
		self.GH.append(self.GH[-1] + self.I[-1] * self.gamma * frac)

		# Q
		self.IH[-1] += self.Q[-1] * self.gamma * frac
		self.GH[-1] += self.Q[-1] * self.gamma * frac
		self.IN[-1] += self.Q[-1] * self.gamma2 * frac
		self.Q.append(self.Q[-1] * (1 - (self.gamma + self.gamma2) * frac))

		self.GT.append(self.GT[-1])
		self.GV.append(self.GV[-1])

	def test_with_com(self, day, com):
		self.GT[day] += (self.S[day] + self.E[day] + self.I[day] + self.A[day]) * self.t_rate
		com.GT[day] += (com.S[day] + com.E[day] + com.I[day] + com.A[day]) * self.t_rate

		self.Q[day] += self.I[day] * self.t_rate
		com.Q[day] += com.I[day] * self.t_rate

		self.I[day] -= self.I[day] * self.t_rate
		com.I[day] -= com.I[day] * self.t_rate

		self.IN[day] += self.A[day] * self.t_rate
		com.IN[day] += com.A[day] * self.t_rate

		self.G[day] += self.A[day] * self.t_rate
		com.G[day] += com.A[day] * self.t_rate

		self.A[day] -= self.A[day] * self.t_rate
		com.A[day] -= com.A[day] * self.t_rate

	def test(self, day):
		self.GT[day] += (self.S[day] + self.E[day] + self.I[day] + self.A[day]) * self.t_rate

		self.Q[day] += self.I[day] * self.t_rate

		self.I[day] -= self.I[day] * self.t_rate

		self.IN[day] += self.A[day] * self.t_rate

		self.G[day] += self.A[day] * self.t_rate

		self.A[day] -= self.A[day] * self.t_rate


class Commuter:
	def __init__(self, S0):
		self.S = [S0]
		self.E = [0]
		self.I = [0]
		self.A = [0]
		self.IH = [0]
		self.IN = [0]
		self.D = [0]
		self.R = [0]
		self.Q = [0]
		self.G = [0]
		self.GH = [0]
		self.GT = [0]
		self.GV = [0]
		self.GI = [0]

	def pop(self, day):
		return self.S[day] + self.E[day] + self.I[day] + self.A[day] + self.IH[day] + self.IN[day] + self.R[day] + \
		       self.D[day] + self.Q[day]

	def sim(self, day):
		self.S.append(self.S[-1])
		self.E.append(self.E[-1])
		self.I.append(self.I[-1])
		self.A.append(self.A[-1])


def simulate(testing, vaccinating, ct_rate, cv_rate, st_rate, sv_rate, fix_s_beta, day_frac,County):
	school = Node(County, '2N', '2N/Population1.csv', 0, st_rate, sv_rate)
	city = Node(County, '2N', '2N/Population1.csv', 5, ct_rate, cv_rate)
	city.S[0] *= 1.5
	city.eta *= 1.5
	'''if County=='IL-Cook':
		school.S[0] = 50000
		school.n_0 = school.S[0]/school.eta'''
	school.S[0] = 10000
	school.n_0 = school.S[0]/school.eta
	commuter = Commuter(school.S[-1] * 0.8)
	school.commute_out(commuter)

	school.beta *= school_beta_multiplier
	city.beta *= city_beta_multiplier

	days = 150

	for i in range(days * 2):
		if i % 2 == 0:  # day time
			city.sim(i + 1, False, day_frac)
			school.sim_with_com(i + 1, commuter, fix_s_beta, day_frac)
			if testing:
				city.test(i + 1)
				school.test_with_com(i + 1, commuter)
			if vaccinating:
				city.vac(i + 1)
				school.vac_with_com(i + 1, commuter)

		else:  # night time
			city.sim_with_com(i + 1, commuter, False, 1 - day_frac)
			school.sim(i + 1, fix_s_beta, 1 - day_frac)

	return city, school, commuter



def school_testing_cost1(cvr,ctr,tc,dtf,county,hc1,hc2,hc3):
    t_low, t_up, t_step = 0, 1, 0.025
    t_rates = np.arange(t_low, t_up + t_step, t_step)
    days = np.arange(0, 150.5, 1)
    cols = ['save \\ test rate']
    cols.extend(t_rates)
    test_cost = tc
    hosp_cost = [hc1,hc2,hc3]
	# normal vaccination rate
    cities = []
    schools = []
    commuters = []
    for t_rate in t_rates:
        city, school, commuter = simulate(testing=True, vaccinating=True, ct_rate=ctr,
		                                  cv_rate=cvr, st_rate=t_rate, sv_rate=cvr,
		                                  fix_s_beta=fix_school_beta, day_frac=dtf,County = county)
        cities.append(city)
        schools.append(school)
        commuters.append(commuter)
    
    base_GI = schools[0].GI[-1] + commuters[0].GI[-1]
    base_GT = schools[0].GT[-1] + commuters[0].GT[-1]
    y1 = [((base_GI - schools[i].GI[-1] - commuters[i].GI[-1]) * hosp_cost[0] -
	       (schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost) / 1000000
	      for i in range(len(t_rates))]
    y1 = np.asarray(y1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_rates,y=y1,mode ="lines",name="saving w/ 5500 / infection",fill = 'tonexty'))
    y2 = [((base_GI - schools[i].GI[-1] - commuters[i].GI[-1]) * hosp_cost[1] -
	       (schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost) / 1000000
	      for i in range(len(t_rates))]
    y2 = np.asarray(y2)
    fig.add_trace(go.Scatter(x=t_rates,y=y2,mode ="lines",name="saving w/ 8500 / infection",fill ='tonextx'))
    y3 = [((base_GI - schools[i].GI[-1] - commuters[i].GI[-1]) * hosp_cost[2] -
	       (schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost) / 1000000
	      for i in range(len(t_rates))]
    y3 = np.asarray(y3)
    fig.add_trace(go.Scatter(x=t_rates,y=y3,mode ="lines",name="saving w/ 11600 / infection",fill = 'tonextx'))
    y4 = [(schools[i].GT[-1] + commuters[i].GT[-1]) * test_cost / 1000000
	                       for i in range(len(t_rates))]
    y4 = np.asarray(y4)
    fig.add_trace(go.Scatter(x=t_rates,y=y4,mode ="lines",name="testing cost"))
    fig.add_shape(type='line',
                x0=0,
                y0=0,
                x1=1,
                y1=0,
                line=dict(color='Red',),
                xref='x',
                yref='y'
)
    fig.update_traces(hoverinfo='text+name', mode='lines')
    fig.update_layout(
    autosize=True,
    #width=900,
    #height=650,
    yaxis = dict( 
       #range = [0,30] ,
       rangemode="tozero",
        autorange=True,
        title_text='million',
        titlefont=dict(size=10),
    ),
    xaxis = dict(
        title_text= "testing rate",
        autorange=True,
        titlefont=dict(size=10),
        zeroline = True
    ),
    )
    fig.update_layout(legend = dict(x=0,y=-0.5))
    return fig

def compare(inp,co,t_rates,cvr,dn):
    
    city, school, commuter = simulate(testing = True, vaccinating = True, ct_rate = 0.02,
                                          cv_rate = cvr,st_rate = t_rates,sv_rate = cvr,
                                          fix_s_beta = fix_school_beta,day_frac = dn,County = co)
 
    
    days = np.arange(0, 150.05, 1)
    

    I_combined = [school.I[t] + commuter.I[t] for t in range(len(school.I))]
    I_combined = I_combined[::2]
    IH_combined = [school.IH[t] + commuter.IH[t] for t in range(len(school.IH))]
    IH_combined = IH_combined[::2]
    GH_combined = [school.GH[t] + commuter.GH[t] for t in range(len(school.GH))]
    GH_combined = GH_combined[::2]       
    df_I = {'days':days,'Infections':I_combined}
    df_I = pd.DataFrame(df_I)
    df_IH = {'days':days,'Hospitalization':IH_combined}
    df_IH = pd.DataFrame(df_IH)
    df_GH = {'days':days,'CumHosp':GH_combined,'Hospitalization':IH_combined,'Infections':I_combined,'t_rate':t_rates}
    df_GH = pd.DataFrame(df_GH)
    #df_GH['CumHosp'] = df_GH['CumHosp'].diff()
    #fig = px.line(df_I,x = df_I['days'],y = df_I['Infections'])
    fig2 = go.Figure(go.Scatter(x=df_GH['days'],y=df_GH['Hospitalization'],name = 'Active Hospitilization'))
    for i in inp:
        if i == 'I':
           fig2.add_trace(go.Scatter(x=df_GH['days'],y=df_GH['Infections'],name = 'Active Infections')) 
        if i == 'GH':
            fig2.add_trace(go.Scatter(x=df_GH['days'],y=df_GH['CumHosp'],name = 'Cummulative Hospitilization'))
    
    fig2.update_traces(hoverinfo='text+name', mode='lines')
    fig2.update_layout(
    autosize=True,
    #width=900,
    #height=650,
    yaxis = dict( 
       #range = [0,100] ,
       #rangemode="tozero",
        autorange=True,
        title_text='Cases',
        titlefont=dict(size=10),
    ),
    xaxis = dict(
        title_text= "Days",
        autorange=True,
        #range = [0,150],
        titlefont=dict(size=10),
        zeroline = True
    ),
    )
    fig2.update_layout(legend = dict(x=0,y=-0.4))
    return fig2

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1('Results'),
            html.Br(),
            html.H4('Choose the figure'),
            dcc.Dropdown(id = 'resfigure',options = [{'label':'figure1','value':'fig'},{'label':'figure2','value':'fig2'}],value = 'fig',style = {'width':'70%'}),
            dcc.Graph(id='fig2',figure = compare('IH','IL-Cook',0.2,0.0015,1/2)),
            html.Br(),
            #dcc.Graph(id="fig",figure = school_testing_cost1(0.0015,0.02,25,1 / 2,'IL-Cook',5500,8500,11600)),
        ], className="six columns"),

        html.Div([
            html.H1('Tweek the Data'),
            dbc.InputGroup(
               [
                   dbc.InputGroupAddon("County"),
                   dcc.Dropdown(id='County',
            options=[
                     {'label': 'IL-Cook', 'value': 'IL-Cook'},
                     {'label':'TX-Dallas','value':'TX-Dallas'},
                     {'label':'NY-New York','value':'NY-New York'},
                     {'label':'CA-Los Angeles','value':'CA-Los Angeles'},
                     {'label':'GA-Fulton','value':'GA-Fulton'},
                     {'label':'MN-Hennepin','value':'MN-Hennepin'},
                     {'label':'NV-Clark','value':'NV-Clark'},
                     {'label':'NJ-Hudson','value':'NJ-Hudson'},
                     {'label':'NJ-Bergen','value':'NJ-Bergen'},
                     {'label':'LA-Jefferson','value':'LA-Jefferson'},
                     {'label':'OH-Franklin','value':'OH-Franklin'},
                     {'label':'PA-Philadelphia','value':'PA-Philadelphia'},
                     {'label':'NC-Mecklenburg','value':'NC-Mecklenburg'},
                     {'label':'TN-Shelby','value':'TN-Shelby'},
                     {'label':'WI-Milwaukee','value':'WI-Milwaukee'},
                     {'label':'VA-Fairfax','value':'VA-Fairfax'},
                     {'label':'AZ-Maricopa','value':'AZ-Maricopa'},
                     {'label':'CA-Riverside','value':'CA-Riverside'},
                     {'label':'FL-Broward','value':'FL-Broward'},
                     {'label':'FL-Miami-Dade','value':'FL-Miami-Dade'},
                     {'label':'MA-Middlesex','value':'MA-Middlesex'},
                     {'label':'TX-Harris','value':'TX-Harris'},
                     {'label':'UT-Salt Lake','value':'UT-Salt Lake'},
                     
                     
            ],
            optionHeight=35,                   
            value='IL-Cook',                    
            disabled=False,                     
            multi=False,                        
            searchable=True,                    
            search_value='',                    
            placeholder='Choose County...',     
            clearable=True,                     
            style={'width':"90%"}          
            ),
               ],
               style={'margin-top':'30px', 'width': '50%', 'float': 'left'},
           ),
        dbc.InputGroup(
               [

                   dbc.Label("Day_Night_Fraction"),
                   dcc.Input(id="D/N",type='number',placeholder="D/N...",
            min=0, max=1, step=0.1,style = {'width':'20%'},value = 1 / 2
        ),
               ],
               className="mb-3",
               style={'margin-top':'20px','width': '80%', 'float': 'left'},
       ),
        dbc.InputGroup(
               [
                   dbc.Label("City Testing Rate"),
                    dcc.Input(id="City_Testing_Rate",type='number',placeholder="testing rate...",
            min=0, max=1, step=0.01,style = {'width':'23%'},
            value=0.02),

               ],
               className="mb-3",
               style={'margin-top':'20px','width': '70%', 'float': 'left'},
       ),
        dbc.InputGroup(
               [
                   dbc.Label("Vaccination Rate"),
                    dcc.Input(id="Vaccination_Rate",type='number',placeholder="vaccination rate...",
            min=0, max=1, step=0.001,style = {'width':'23%'}
            ,value = 0.003),

               ],
               className="mb-3",
               style={'margin-top':'20px','width': '70%', 'float': 'left'},
       ),dbc.InputGroup(
               [
                   dbc.Label("Testing Cost"),
                    dcc.Input(id="Testing_Cost",type='number',placeholder="Testing rate...",
            min=0, max=100, step=1,style = {'width':'23%'}
            ,value = 25),

               ],
               className="mb-3",
               style={'margin-top':'20px','width': '70%', 'float': 'left'},
       ),dbc.InputGroup(
               [
                   dcc.Checklist(id='checkbox1',options = [{'label':'Active Infection','value':'I'},
                                                     {'label':'Active hospitilization','value':'IH'},
                                                     {'label':'Cummulative hospitilization','value':'GH'}],value = ['IH','I','GH'])
               ],
               className="mb-3",
               style={'margin-top':'20px','width': '70%', 'float': 'left'},
       ),dbc.InputGroup(
               [
                   html.Label('Testing Rate :'),
            dcc.Input(id="Testing_Rate",type='number',placeholder="Testing rate...",
            min=0, max=1, step=0.1,style = {'width':'23%'}
            ,value = 0.2),
               ],
               className="mb-3",
               style={'margin-top':'20px','width': '70%', 'float': 'left'},
       ),
           dbc.InputGroup(
               [
                   html.Label('Hospitalization Cost :'),
        dcc.Input(id="HC1",type='number',placeholder="Hospitalization cost...",
            min=1, max=1000000, step=1,style = {'width':'18%'},value = 5500),
        dcc.Input(id="HC2",type='number',placeholder="Hospitalization cost...",
            min=1, max=1000000, step=1,style = {'width':'18%'},value = 8500),
        dcc.Input(id="HC3",type='number',placeholder="Hospitalization cost...",
            min=1, max=1000000, step=1,style = {'width':'18%'},value = 11600)
        
               ],
               className="mb-3",
               style={'margin-top':'20px','width': '70%', 'float': 'left'},
       )
        ], className="six columns"),
    ], className="row")
])

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

'''@app.callback(
    Output('fig', 'figure'),
    Input('vr', 'value'),
    Input('tr', 'value'),
    Input('tc', 'value'),
    Input('D/N', 'value'),
    Input('County','value'),
    Input('HC1','value'),
    Input('HC2','value'),
    Input('HC3','value'),
    Input('checkbox1','value'),
    Input('trs','value'))
def update_figure(cvr,ctr,tc,t,county,hc1,hc2,hc3):
    fig = school_testing_cost1(cvr,ctr,tc,t/(t+1),county,hc1,hc2,hc3)
    fig.update_layout(transition_duration=500)
    return fig'''
@app.callback(Output('fig2','figure'),
              Input('resfigure','value'),
              Input('checkbox1','value'),
              Input('County','value'),
              Input('Testing_Rate', 'value'),
              Input('Vaccination_Rate', 'value'),
              Input('City_Testing_Rate', 'value'),
              Input('D/N', 'value'),
              Input('Testing_Cost', 'value'),
              Input('HC1','value'),
              Input('HC2','value'),
              Input('HC3','value'),
              )
def update_figure2(pic,inp,county,t_rate,cvr,ctr,t,tc,hc1,hc2,hc3):
    if pic == 'fig':
        fig = compare(inp,county,t_rate,cvr,t/(t+1))
        fig.update_layout(transition_duration=100)
    else:
        fig = school_testing_cost1(cvr,ctr,tc,t/(t+1),county,hc1,hc2,hc3)
        fig.update_layout(transition_duration=500)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)