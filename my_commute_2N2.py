import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from SIRfunctions import SEIARG, SEIARG_fixed, computeBeta_combined
from matplotlib.colors import LogNorm
from matplotlib.dates import DateFormatter
import os

from plotly.subplots import make_subplots
import plotly.graph_objects as go

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


def simulate(testing, vaccinating, ct_rate, cv_rate, st_rate, sv_rate, fix_s_beta, day_frac):
	school = Node('school', '2N', '2N/Population.csv', 0, st_rate, sv_rate)
	city = Node('IL-Cook', '2N', '2N/Population.csv', 5, ct_rate, cv_rate)
	city.S[0] *= 1.5
	city.eta *= 1.5
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


	
def compare():
    
    t_low,t_up,t_step = 0, 1, 0.2
    t_rates = np.arange(t_low, t_up + t_step, t_step)
    cities = []
    schools = []
    commuters = []
    for t_rate in t_rates:
        city, school, commuter = simulate(testing = True, vaccinating = True, ct_rate = city_test_rate,
                                          cv_rate = city_vaccine_rate,st_rate = t_rate,sv_rate = city_vaccine_rate,
                                          fix_s_beta = fix_school_beta,day_frac = daytime_fraction)
        cities.append(city)
        schools.append(school)
        commuters.append(commuter)
    
    days = np.arange(0, 151, 1)
    
    for i in range(len(t_rates)):
        I_combined = [schools[i].I[t] + commuters[i].I[t] for t in range(len(schools[i].I))]
        I_combined = I_combined[::2]
        
    
    df = {'days':days,'Infections':I_combined}
    df = pd.DataFrame(df)
    
    plt.plot(df['days'], df['Infections'])
    plt.show()
    
    
    return


def main():
    compare()
    
    return


if __name__ == "__main__":
	main()
