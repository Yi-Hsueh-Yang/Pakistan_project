# -*- coding: utf-8 -*-
"""
@author: aliceji

The following data is collected from the World Bank country dataset: https://data.worldbank.org/indicator/SM.POP.NETM?locations=PK
This dataset has a timespan from 1970 to 2020. For analysis purpose, we primarily
focuson 2001 onward and this contains features ranging from energy consumption behaviours 
to demographical shift. We aim to fit a multiiple linear regression based on these feasures 
to better understand climate-induced migration patterns. 

 
"""

import pandas as pd
import numpy as np
import csv
import datetime 
from sklearn import linear_model
import matplotlib.pyplot as plt




df_new= pd.read_csv("migration.csv")
# Data cleanup and rename columns
df_new['Date'] = '20'+df_new['Date'].apply(lambda x: str(x).split('/')[-1])


df_new.columns = ['Date', 'Fossile_use', 'ELec_coal',
              'CO2_total','CO2_per_capita',
              'GDP_per_capita','population',
              'annual_change_ppl',
              'net_m_per1000','Net_Migration_Rate',
              'hunger_percentage','clean_water_percentage']

#drop missing values
df_dropped = df_new.dropna()

X = df_dropped[['Fossile_use', 'ELec_coal',
              'CO2_total','CO2_per_capita',
              'GDP_per_capita','population',
              'annual_change_ppl',
              'hunger_percentage','clean_water_percentage']]

Y1 = df_dropped['Net_Migration_Rate']

#fit linear model
regr = linear_model.LinearRegression()
regr.fit(X, Y1)

model_result = linear_model.LinearRegression().fit(X, Y1)
def get_data():
  return df_dropped


def visualize():
  
  plt.plot(df_dropped['Date'], df_dropped['Net_Migration_Rate'], color="blue", linewidth=3)
  plt.title("Net Migration Scale Prediction")
  plt.show()
  
  


def get_model_result():
  print("Model result for migration scale prediction")
  print(f"intercept: {model_result.intercept_}")
  print(f"slope: {model_result.coef_}")
  
