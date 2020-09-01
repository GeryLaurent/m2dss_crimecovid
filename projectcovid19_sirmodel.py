# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 10:27:14 2020

@author: Gery
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit
from scipy.integrate import odeint

########################################
# Data collection & Data management
########################################

# Data import with daily number of cases, death and recovered patients
df = pd.read_csv("https://raw.githubusercontent.com/opencovid19-fr/data/master/dist/chiffres-cles.csv")
dfFrance = df[df['granularite'] == "pays"]
dfFranceFiltered = dfFrance[['date','cas_confirmes','deces','gueris','source_nom']]

# Regroupement des informations des différentes sources par date
dfFranceFused = dfFranceFiltered.groupby('date').max().reset_index().drop(['source_nom'], axis=1)

# Gestion des données manquantes: interpolation
dfFranceInterpolated = dfFranceFused.interpolate().round(decimals=0).fillna(0)

# Reconstruction des données SIR pour le modèle simplifié
# Population totale de la France: 66,99 millions
# S = Susceptible: Population totale - gueris - deces
# I = Infecté: cas confirmés - gueris - deces
# R = Retiré: gueris+deces

dfFranceInterpolated['pop'] = 64890000 # metropolitan France population
dfFranceInterpolated['S'] = dfFranceInterpolated['pop'] - dfFranceInterpolated['deces'] - dfFranceInterpolated['gueris']
dfFranceInterpolated['I'] = dfFranceInterpolated['cas_confirmes'] - dfFranceInterpolated['deces'] - dfFranceInterpolated['gueris']
dfFranceInterpolated['R'] = dfFranceInterpolated['deces'] + dfFranceInterpolated['gueris']

dfFranceInterpolated['time_all'] = np.arange(len(dfFranceInterpolated))

data_all = dfFranceInterpolated[['I','R']].to_numpy().transpose()
day_all = dfFranceInterpolated['date']
day_all_ticks = day_all[::14]
t_all = dfFranceInterpolated['time_all'] 

########################################
# Data visualisation
########################################

# Plot I,R
plt.figure(figsize=[14,4])
plt.plot(day_all, data_all[0].tolist(), '-')
plt.plot(day_all, data_all[1].tolist(), '-') 
plt.legend(["I","R"])
plt.ylabel("population")
plt.xlabel("Time (Day)")
plt.xticks(day_all_ticks, rotation=300)
plt.vlines("2020-03-11",ymin=0, ymax=max(data_all[0]), linestyles="dashed",colors='grey')
plt.text("2020-03-14", max(data_all[0]), "lockdown start", color='grey')
plt.vlines("2020-05-11",ymin=0, ymax=max(data_all[0]), linestyles="dashed",colors='grey')
plt.text("2020-05-14", max(data_all[0]), "lockdown end", color='grey')
plt.show()


# Séparation des données en 2 phases:
    # pré-confinement: 24 janvier au 10 mars 2020
 
dfPreConf = dfFranceInterpolated[dfFranceInterpolated['date'] < "2020-03-11"].reset_index()
dfPreConf['time'] = np.arange(len(dfPreConf))

    # pré-confinement et confinement: 24 janvier au 10 mai 2020
dfConf = dfFranceInterpolated[(dfFranceInterpolated['date'] < "2020-05-11")].reset_index()
dfConf['time'] = np.arange(len(dfConf))


# The SIR model differential equations.
def deriv(y, t, N, beta, gamma): # ODE function
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def Model(beta, gamma, t): # Model function
    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    return odeint(deriv, y0, t, args=(N, beta, gamma))

def objective(params, t, data):
    vals = params.valuesdict()
    ndata, nx = data.shape
    resid = 0.0*data[:]
    # make residual per data set
    for i in range(ndata):
        resid[i, :] = data[i, :] - np.delete(Model(vals['beta_1'],vals['gamma_1'],t), 0, axis=1).transpose()[i]
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()



########################################
# Pre lockdown period
########################################

data = dfPreConf[['I','R']].to_numpy().transpose()

# Initiate parameters
# create 2 sets of parameters, one per data set
fit_params = Parameters()
for iy, y in enumerate(data):
    fit_params.add( 'beta_%i' % (iy+1), value=0.2, min=0.0,  max=1.0)
    fit_params.add( 'gamma_%i' % (iy+1), value=0.4, min=0.0,  max=1.0)
# but now constrain all values of beta and gamma to have the same value
fit_params['beta_%i' % 2].expr='beta_1'
fit_params['gamma_%i' % 2].expr='gamma_1'


# Total population, N.
N = dfPreConf['pop'][0]
# Initial number of infected and recovered individuals, I0 and R0.
I0 = dfPreConf['I'][0]
R0 = dfPreConf['R'][0]
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# A grid of time points (in days)
t = dfPreConf['time']    
# run the global fit to all the data sets
result = minimize(objective, fit_params, args=(t, data))
report_fit(result.params)
R_0 = result.params['beta_1'].value / result.params['gamma_1'].value
print(R_0)

t_future  = np.linspace( 0, 300, 300) # time grid of 300 days

# Plot
plt.figure()
y_fit_I = np.delete(Model(result.params['beta_1'].value,result.params['gamma_1'].value,t), 0, axis=1).transpose()[0]
y_fit_R = np.delete(Model(result.params['beta_1'].value,result.params['gamma_1'].value,t), 0, axis=1).transpose()[1]
plt.plot(t, data[0].tolist(), '.')
plt.plot(t, data[1].tolist(), '.') 
plt.plot(t, y_fit_I, '-')
plt.plot(t, y_fit_R, '-')
plt.legend(["I","R", "fitted I","fitted R"])
plt.ylabel("population")
plt.xlabel("Time (Day)")
plt.show()

# Plot over 300 days
plt.figure()
y_fit_S = Model(result.params['beta_1'].value,result.params['gamma_1'].value,t_future).transpose()[0]
y_fit_I = np.delete(Model(result.params['beta_1'].value,result.params['gamma_1'].value,t_future), 0, axis=1).transpose()[0]
y_fit_R = np.delete(Model(result.params['beta_1'].value,result.params['gamma_1'].value,t_future), 0, axis=1).transpose()[1]
plt.plot(t_all, data_all[0].tolist(), '.')
plt.plot(t_all, data_all[1].tolist(), '.') 
plt.plot(t_future, y_fit_I, '-')
plt.plot(t_future, y_fit_R, '-')
plt.plot(t_future, y_fit_S, '-')  
plt.legend(["I","R", "fitted I","fitted R","fitted S"])
plt.ylabel("population")
plt.xlabel("Time (Day)")
plt.show()

########################################
# Pre lockdown and lockdown period
########################################

data = dfConf[['I','R']].to_numpy().transpose()
# Total population, N.
N = dfConf['pop'][0]
# Initial number of infected and recovered individuals, I0 and R0.
I0 = dfConf['I'][0]
R0 = dfConf['R'][0]
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# A grid of time points (in days)
t = dfConf['time_all']
t_future  = np.linspace( 0, 300, 300) # time grid of 300 days

# run the global fit to all the data sets
result = minimize(objective, fit_params, args=(t, data))
report_fit(result.params)
R_0 = result.params['beta_1'].value / result.params['gamma_1'].value
print(R_0)

# Plot
plt.figure()
y_fit_I = np.delete(Model(result.params['beta_1'].value,result.params['gamma_1'].value,t), 0, axis=1).transpose()[0]
y_fit_R = np.delete(Model(result.params['beta_1'].value,result.params['gamma_1'].value,t), 0, axis=1).transpose()[1]
plt.plot(t, data[0].tolist(), '.')
plt.plot(t, data[1].tolist(), '.') 
plt.plot(t, y_fit_I, '-')
plt.plot(t, y_fit_R, '-')
plt.legend(["I","R", "fitted I","fitted R"])
plt.ylabel("population")
plt.xlabel("Time (Day)")
plt.show()

# Plot over 300 days
plt.figure()
y_fit_S = Model(result.params['beta_1'].value,result.params['gamma_1'].value,t_future).transpose()[0]
y_fit_I = np.delete(Model(result.params['beta_1'].value,result.params['gamma_1'].value,t_future), 0, axis=1).transpose()[0]
y_fit_R = np.delete(Model(result.params['beta_1'].value,result.params['gamma_1'].value,t_future), 0, axis=1).transpose()[1]
plt.plot(t_all, data_all[0].tolist(), '.')
plt.plot(t_all, data_all[1].tolist(), '.') 
plt.plot(t_future, y_fit_I, '-')
plt.plot(t_future, y_fit_R, '-')
plt.plot(t_future, y_fit_S, '-')  
plt.legend(["I","R", "fitted I","fitted R","fitted S"])
plt.ylabel("population")
plt.xlabel("Time (Day)")
plt.show()