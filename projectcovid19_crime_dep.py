# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 01:56:44 2020

@author: Gery
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose 
from pmdarima import auto_arima

########################################
# Data collection & Data management
########################################

xlFile = pd.ExcelFile("https://www.data.gouv.fr/fr/datasets/r/fdf5afbf-ed3c-4c54-a4f0-3581c8a1eca4")
xlSheetDep = xlFile.sheet_names[2:-11]

dfCrimeDict = pd.read_excel(xlFile, sheet_name=xlSheetDep)

tx_variation0420 = []
for i in range(len(xlSheetDep)): # Loop through department and return crime percentage change for April 2020
    dfCrime = dfCrimeDict[xlSheetDep[i]]
    dfCrime.sum(axis=0)
    
    dfCrime_transposed = dfCrime.transpose()
    dfCrime_transposed.columns = dfCrime_transposed.iloc[0]
    dfCrime_transposed = dfCrime_transposed.drop(['Index','libellé index'])
    dfCrime_transposed = dfCrime_transposed.astype(int)
    dfCrime_transposed['total'] = dfCrime_transposed.sum(axis=1)
    
    dfCrime_total = dfCrime_transposed['total'].reset_index().rename(columns={'index':'mois'})
    
    dfCrime_sup2010 = dfCrime_total[dfCrime_total['mois'] >= '2010-01'].sort_values(['mois'])
    
    dfCrime_2010_2019 = dfCrime_sup2010[dfCrime_sup2010['mois'] < '2020-01-01']
    timeseries_10_19 = dfCrime_2010_2019.set_index('mois')
    timeseries_10_19.index = pd.to_datetime(timeseries_10_19.index, format='%Y_%m')
    
    dfCrime_2019_2020 = dfCrime_sup2010[dfCrime_sup2010['mois'] >= '2019-01-01']
    timeseries_19_20 = dfCrime_2019_2020.set_index('mois')
    timeseries_19_20.index = pd.to_datetime(timeseries_19_20.index, format='%Y_%m')
    
    crime_decomposed = seasonal_decompose(timeseries_10_19['total'], model='multiplicative')
    
    # Fit auto_arima function to dataset 
    stepwise_fit = auto_arima(timeseries_10_19['total'], start_p = 1, start_q = 1, 
                              max_p = 3, max_q = 3, m = 12, 
                              start_P = 0, seasonal = True, 
                              d = None, D = 1, trace = True, 
                              error_action ='ignore',
                              suppress_warnings = True,
                              stepwise = True)
    
    ### FIT TO DATASET
    
    # Split data into train / test sets 
    train = timeseries_10_19.iloc[:len(timeseries_10_19)-12] 
    test = timeseries_10_19.iloc[len(timeseries_10_19)-12:] # set one year(12 months) for testing 
      
    # Fit a SARIMAX on the training set 
    from statsmodels.tsa.statespace.sarimax import SARIMAX 
      
    model = SARIMAX(train['total'],  
                    order = stepwise_fit.order,  
                    seasonal_order = stepwise_fit.seasonal_order) 
      
    result = model.fit() 
    
    ### PREDICTIONS AGAINST TEST
    
    start = len(train) 
    end = len(train) + len(test) - 1
      
    # Predictions for 12 months against the test set 
    predictions = result.predict(start, end, 
                                 typ = 'levels').rename("Predictions") 
      
    ### FORECASTING USING ARIMA
    
    # Train the model on the full dataset 
    model = SARIMAX(timeseries_10_19['total'],  
                    order = (1,0,1),  
                    seasonal_order =(0, 1, [1, 2], 12)) 
      
    result = model.fit() 
      
    # Forecast for the next 3 years 
    forecast = result.predict(start = len(timeseries_10_19),  
                              end = (len(timeseries_10_19)-1) + 7,  
                              typ = 'levels').rename('Forecast') 
      
    # Plot the forecast values 
    timeseries_19_20['total'][12:].plot(figsize = (12, 5), legend = True) 
    forecast.plot(legend = True)
    
    # Calculate % differences
    
    dfDiff = pd.merge(timeseries_19_20[12:],forecast, how='left', left_index=True, right_index=True)
    dfDiff['tx_variation'] = (dfDiff['total']-dfDiff['Forecast']) / dfDiff['Forecast'] * 100
    
    tx_variation0420.append(round(dfDiff['tx_variation'][3],2))

# Make dataframe
df_variation_dep = pd.DataFrame(list(zip(xlSheetDep,tx_variation0420)),columns=['dep','tx_variation'])
df_variation_dep.to_csv("df_variation_dep0420.csv")
    
## MAKE CHOROPLETH MAP
import json
with open("departements.geojson") as f:
    franceMap = json.load(f)
   
sortedMap = dict(franceMap)
sortedMap['features'] = sorted(franceMap['features'], key=lambda x: x['properties']['code'])

import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
fig = px.choropleth_mapbox(df_variation_dep, geojson=sortedMap, locations='dep', featureidkey = 'properties.code', color='tx_variation',
                           color_continuous_scale=['#000066','#0000ff','#0066ff','#3399ff','#ffffff'], range_color=[-80,0],
                           mapbox_style="carto-positron",
                           zoom=5.4, center = {"lat": 46.5, "lon": 1},
                           opacity=0.7,
                           labels={'dep':'code du départment','tx_variation':'taux de variation de la criminalité'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, coloraxis_colorbar=dict(ticksuffix=" %"))
fig.show()