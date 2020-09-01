# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:44:08 2020

@author: Gery
"""

import datetime
from datetime import datetime
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly

########################################
# Data collection & Data management
########################################

with open("departements.geojson") as f:
    franceMap = json.load(f)
   
sortedMap = dict(franceMap)
sortedMap['features'] = sorted(franceMap['features'], key=lambda x: x['properties']['code'])


df = pd.read_csv("https://raw.githubusercontent.com/opencovid19-fr/data/master/dist/chiffres-cles.csv")
dfDepartment = df[df['maille_code'].str.contains("DEP")]
dfDepartment['code_dep'] = dfDepartment['maille_code'].str.split('-').str[1]
dfDepartmentMet = dfDepartment[~dfDepartment['code_dep'].isin(['971','972','973','974','976']) ] # Remove overseas departments


# Données hospitalières par département [total]
dfHospit = pd.read_csv("https://www.data.gouv.fr/en/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7",sep=";")
dfHospitMet = dfHospit[~dfHospit['dep'].isin(['971','972','973','974','976',None]) ].dropna()
dfHospitMet['jour'] = dfHospitMet['jour'].apply(lambda x: x if x.startswith('2020') else datetime.strptime(x, '%d/%m/%Y').strftime('%Y-%m-%d')) # Date conversion to correct datetime errors in the data

# Données nombre lit réa par département
dfLit = pd.read_csv("lit_rea2018.csv")
dfLitMet = dfLit[~dfLit['dep'].isin(['971','972','973','974','976','FR','MET',None]) ]
dfLitMet['nb_lit'] = dfLitMet['nb_lit'].astype(int)


# Nombre de patients hospitalisés, en réanimation pour Covid-19
dfHospitMetCleaned = dfHospitMet[dfHospitMet['sexe'] == 0].drop(['sexe','rad','dc','hosp'], axis=1).sort_values(by=['dep','jour'])

dfReaLit = pd.merge(dfHospitMetCleaned, dfLitMet, how='left', on=['dep'])
dfReaLit['tx_occ'] = dfReaLit['rea'] / dfReaLit['nb_lit']
dfReaLit['pt_occ'] = dfReaLit['tx_occ']*100

dfReaLit_map = dfReaLit.filter(['dep','pt_occ','jour'])
listRange = np.arange(0, len(dfReaLit_map['jour'].unique()), 7).tolist()
dfReaLit_mapSliced = dfReaLit_map.groupby('dep').nth(listRange).reset_index()

########################################
# Data visualisation
########################################

fig = px.choropleth_mapbox(dfReaLit_mapSliced, geojson=sortedMap, locations='dep', featureidkey = 'properties.code', color='pt_occ',
                           animation_frame='jour',
                           color_continuous_scale=['#ffffff','#ff0000',"#990000",'#420000'], range_color=[0,300],
                           mapbox_style="carto-positron",
                           zoom=5, center = {"lat": 46.5, "lon": 1},
                           opacity=0.9,
                           labels={'dep':'code du départment','pt_occ':"pourcentage d'occupation des lits de réanimation"}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                  coloraxis_colorbar=dict(ticksuffix=" %"))
fig.layout.coloraxis.showscale = True
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 4000
fig.layout.updatemenus[0].buttons[0].args[1]["mode"] = None
fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 2000
fig.layout.sliders[0].pad.t = 10
fig.layout.updatemenus[0].pad.t= 10
fig.write_html("D:\\CovidProject\\reaAnimated.html") # Save animation into .html file.

